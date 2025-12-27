import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm

# These come from the original repo:
# - plot_ci
# - plot_ci_plus_heatmap
from utils import plot_ci, plot_ci_plus_heatmap

from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# Reproducibility
# -----------------------------
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


# -----------------------------
# Helper: OLMo-2 wrapper
# -----------------------------
class Olmo2Helper:
    """
    Minimal helper to:
      - load tokenizer/model from HF hub or local path
      - return hidden states for all layers
      - expose "unemb" = final_norm -> lm_head (like your llama code)
    """

    def __init__(
        self,
        model_id_or_path: str,
        revision: Optional[str] = None,
        load_in_8bit: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_id_or_path = model_id_or_path
        self.revision = revision

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path,
            revision=revision,
            use_fast=True,
        )

        if load_in_8bit:
            # Needs bitsandbytes installed and GPU support
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id_or_path,
                revision=revision,
                device_map="auto",
                load_in_8bit=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id_or_path,
                revision=revision,
                device_map="auto",
                torch_dtype=dtype,
            )

        self.model.eval()

        # Output embeddings (lm_head)
        self.lm_head = self.model.get_output_embeddings()

        # Find final norm (OLMo2 base model usually has model.norm)
        base = getattr(self.model, "model", None) or getattr(self.model, "base_model", None) or self.model
        if hasattr(base, "norm"):
            self.final_norm = base.norm
        else:
            # fallback attempts (in case a future HF wrapper changes names)
            for cand in ["final_layer_norm", "final_layernorm", "ln_f"]:
                if hasattr(base, cand):
                    self.final_norm = getattr(base, cand)
                    break
            else:
                raise AttributeError(
                    "Could not find a final norm module on this checkpoint. "
                    "Tried: model.norm, final_layer_norm, final_layernorm, ln_f."
                )

        # Your original script builds "unemb" = norm + lm_head
        self.unemb = nn.Sequential(self.final_norm, self.lm_head)

    @torch.no_grad()
    def latents_all_layers(self, prompt: str) -> torch.Tensor:
        """
        Returns: latents of shape [L, B, T, H] for L transformer layers.
        We drop the embedding output state to match typical "layer indexing".
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple: (emb, layer1, ..., layerN)
        hs = hs[1:]             # drop embedding output
        return torch.stack(hs, dim=0)  # [L, B, T, H]


# -----------------------------
# Token utilities (OLMo-2-safe)
# -----------------------------
def process_tokens(token_str: str, tokenizer) -> List[int]:
    """
    Convert a string into token ids, tokenizer-agnostic.
    We keep unique ids because later you sum probs over ids anyway.
    """
    ids = tokenizer.encode(token_str, add_special_tokens=False)
    return list(set(ids))


def compute_entropy(probas: torch.Tensor) -> torch.Tensor:
    # probas: [..., vocab]
    # Avoid log2(0)
    eps = 1e-12
    return (-probas * torch.log2(probas.clamp_min(eps))).sum(dim=-1)


lang2name = {"fr": "Français", "de": "Deutsch", "ru": "Русский", "en": "English", "zh": "中文"}


def build_fewshot_prompt(df: pd.DataFrame, ind: int, k: int, lang1: str, lang2: str) -> Optional[Dict[str, Any]]:
    """
    Mimics your original `sample()` generator but returns a dict or None.

    We create k-1 example lines:
        <Lang1>: "<x>" - <Lang2>: "<y>"
    and one query line with lang2 left open:
        <Lang1>: "<x_q>" - <Lang2>: "

    Then we compute:
        out_token_id: token ids for correct translation (lang2 string)
        latent_token_id: token ids for English string (row['en'])
    We skip cases where out and latent share a token id (for non-English target),
    matching your original intersection-based filtering.
    """
    df = df.reset_index(drop=True)

    if ind < 0 or ind >= len(df):
        return None

    # build prompt by sampling k-1 other rows + the queried row
    temp = df[df.index != ind]
    sample_rows = pd.concat([temp.sample(k - 1, random_state=seed), df[df.index == ind]], axis=0)

    prompt = ""
    out_token_str = None
    in_token_str = None
    latent_token_str = None

    # iterate in order: first k-1 are examples, last is query
    for idx, (_, row) in enumerate(sample_rows.iterrows()):
        if idx < k - 1:
            prompt += f'{lang2name[lang1]}: "{row[lang1]}" - {lang2name[lang2]}: "{row[lang2]}"\n'
        else:
            prompt += f'{lang2name[lang1]}: "{row[lang1]}" - {lang2name[lang2]}: "'
            in_token_str = row[lang1]
            out_token_str = row[lang2]
            latent_token_str = row["en"]  # latent language always English in this script

    if out_token_str is None or in_token_str is None or latent_token_str is None:
        return None

    out_token_id = process_tokens(out_token_str, tokenizer)
    latent_token_id = process_tokens(latent_token_str, tokenizer)

    if len(out_token_id) == 0 or len(latent_token_id) == 0:
        return None

    # Original script: skip if target lang != 'en' and intersection is non-empty
    if lang2 != "en" and len(set(out_token_id).intersection(set(latent_token_id))) > 0:
        return None

    return {
        "prompt": prompt,
        "out_token_id": out_token_id,
        "out_token_str": out_token_str,
        "latent_token_id": latent_token_id,
        "latent_token_str": latent_token_str,
        "in_token_str": in_token_str,
    }


# -----------------------------
# Main config (edit these)
# -----------------------------
input_lang = "fr"
target_lang = "zh"
k_fewshot = 5

# Choose ONE of the following:
USE_HUB_BASE = False

# (A) Base model on the hub:
hub_model_id = "allenai/OLMo-2-1124-7B"
hub_revision = None  # e.g., "main" or a tag/commit if needed

# (B) Your local continual-pretrained HF checkpoint:
local_model_dir = "/scratch/e1583535/llm/nus-olmo/para-only-7B-34B-checkpoints/step8290-unsharded-hf"

# Loading choices:
load_in_8bit = True   # set False if bitsandbytes is unavailable
dtype = torch.bfloat16

# Dataset / outputs:
prefix = "/scratch/e1583535/llm-latent-language/data/langs/"
out_dir = "/scratch/e1583535/llm-latent-language/visuals"

single_token_only = False
multi_token_only = False


# -----------------------------
# Sanity checks
# -----------------------------
if single_token_only and multi_token_only:
    raise ValueError("single_token_only and multi_token_only cannot both be True.")


# -----------------------------
# Load data
# -----------------------------
df_en_in = pd.read_csv(f"{prefix}{input_lang}/clean.csv").reindex()
df_en_tgt = pd.read_csv(f"{prefix}{target_lang}/clean.csv").reindex()

# Rename / merge to get a single df with columns: ['en', input_lang, target_lang]
# - clean.csv format in the repo: 'word_original' and 'word_translation'
#   where word_original is English and word_translation is the target language word.
if input_lang == target_lang:
    df_merged = df_en_tgt.copy()
    df_merged.rename(
        columns={
            "word_original": "en",
            "word_translation": target_lang if target_lang != "en" else "en_tgt",
        },
        inplace=True,
    )
else:
    df_merged = df_en_tgt.merge(df_en_in, on=["word_original"], suffixes=(f"_{target_lang}", f"_{input_lang}"))
    df_merged.rename(
        columns={
            "word_original": "en",
            f"word_translation_{target_lang}": target_lang if target_lang != "en" else "en_tgt",
            f"word_translation_{input_lang}": input_lang if input_lang != "en" else "en_in",
        },
        inplace=True,
    )

# Remove rows where English appears inside the target string (same as your original)
if target_lang != "en":
    drop_idx = []
    for i, row in df_merged.iterrows():
        try:
            if str(row["en"]).lower() in str(row[target_lang]).lower():
                drop_idx.append(i)
        except Exception:
            pass
    if drop_idx:
        df_merged.drop(drop_idx, inplace=True)

df_merged = df_merged.reset_index(drop=True)
print(f"final length of df_merged: {len(df_merged)}")


# -----------------------------
# Load model
# -----------------------------
if USE_HUB_BASE:
    model_id_or_path = hub_model_id
    revision = hub_revision
else:
    model_id_or_path = local_model_dir
    revision = None

olmo = Olmo2Helper(
    model_id_or_path=model_id_or_path,
    revision=revision,
    load_in_8bit=load_in_8bit,
    dtype=dtype,
)

tokenizer = olmo.tokenizer
model = olmo.model
unemb = olmo.unemb

print(f"Loaded model from: {model_id_or_path}")
if revision is not None:
    print(f"  revision: {revision}")


# -----------------------------
# Build U_normalized and avgUU for "energy" (same math as your original)
# -----------------------------
with torch.no_grad():
    # lm_head weight: [vocab, hidden]
    U = list(unemb[1].parameters())[0].detach().cpu().float()
    # final norm weight: typically [hidden]
    weights = list(unemb[0].parameters())[0].detach().cpu().float()
    U_weighted = U.clone()
    U_weighted *= weights.unsqueeze(0)
    U_normalized = U_weighted / ((U_weighted**2).sum(dim=1, keepdim=True)) ** 0.5

    v = U.shape[0]
    avgUU = (((U_normalized.T @ U_normalized) ** 2).sum() / (v**2)) ** 0.5
    print(f"U {U.shape} | norm weights {weights.shape} | avgUU={avgUU.item():.6f}")


# -----------------------------
# Optional: single-token / multi-token filtering (OLMo-2-safe)
# -----------------------------
def num_tokens(s: str) -> int:
    return len(tokenizer.encode(str(s), add_special_tokens=False))


if single_token_only or multi_token_only:
    keep_rows = []
    for i, word in enumerate(df_merged[target_lang].tolist()):
        n = num_tokens(word)
        if single_token_only and n == 1:
            keep_rows.append(i)
        if multi_token_only and n > 1:
            keep_rows.append(i)
    df_merged = df_merged.iloc[keep_rows].reset_index(drop=True)
    print(f"After token-length filtering: {len(df_merged)} rows")


# -----------------------------
# Build dataset (few-shot prompts)
# -----------------------------
dataset: List[Dict[str, Any]] = []
for ind in tqdm(range(len(df_merged)), desc="Building dataset"):
    d = build_fewshot_prompt(df_merged, ind, k=k_fewshot, lang1=input_lang, lang2=target_lang)
    if d is None:
        continue
    dataset.append(d)

df_dataset = pd.DataFrame(dataset)


# -----------------------------
# Output directory (fix absolute-path join issue)
# -----------------------------
# If model_id_or_path is an absolute path, joining it directly would override out_dir.
run_name = os.path.basename(os.path.normpath(model_id_or_path))
save_dir = os.path.join(out_dir, run_name, "translation")
os.makedirs(save_dir, exist_ok=True)

# Save dataset CSV
suffix = ""
if single_token_only:
    suffix = "_single_token"
elif multi_token_only:
    suffix = "_multi_token"

dataset_csv = os.path.join(save_dir, f"olmo2_{input_lang}_{target_lang}_dataset{suffix}.csv")
df_dataset.to_csv(dataset_csv, index=False)
print(f"Saved dataset to: {dataset_csv}")

print(df_dataset.head())


# -----------------------------
# Run analysis (per layer)
# -----------------------------
latent_token_probs = []
out_token_probs = []
entropy = []
energy = []
latents_all = []

for _, d in tqdm(list(enumerate(dataset)), desc="Running model"):
    latents = olmo.latents_all_layers(d["prompt"])  # [L, B, T, H]
    logits = unemb(latents)                         # [L, B, T, V]
    last = logits[:, 0, -1, :].float().softmax(dim=-1).detach().cpu()  # [L, V]

    latent_ids = torch.tensor(d["latent_token_id"], dtype=torch.long)
    out_ids = torch.tensor(d["out_token_id"], dtype=torch.long)

    latent_token_probs.append(last[:, latent_ids].sum(dim=-1))  # [L]
    out_token_probs.append(last[:, out_ids].sum(dim=-1))        # [L]
    entropy.append(compute_entropy(last))                       # [L]

    # store final-position latents [L, H]  (select batch=0, last token)
    lat_last = latents[:, 0, -1, :].float()                 # [L, H]
    latents_all.append(lat_last.detach().cpu().clone())     # save CPU copy

    # energy (FIXED): (V,H) @ (H,L) -> (V,L) -> reduce over V -> [L]
    lat_norm = lat_last
    lat_norm = lat_norm / (((lat_norm**2).mean(dim=-1, keepdim=True)) ** 0.5)
    lat_norm = lat_norm / (lat_norm.norm(dim=-1, keepdim=True) + 1e-12)

    proj = U_normalized.to(lat_norm.device) @ lat_norm.T    # [V, L]
    norm_val = (proj.pow(2).mean(dim=0)).sqrt()             # [L]
    energy.append((norm_val / avgUU.to(lat_norm.device)).detach().cpu())


latent_token_probs = torch.stack(latent_token_probs)  # [N, L]
out_token_probs = torch.stack(out_token_probs)        # [N, L]
entropy = torch.stack(entropy)                        # [N, L]
energy = torch.stack(energy)                          # [N, L]
latents = torch.stack(latents_all)                    # [N, L, H]

print("latents:", latents.shape)
print("latent_token_probs:", latent_token_probs.shape)
print("out_token_probs:", out_token_probs.shape)


# -----------------------------
# Plot 1: probs + entropy heatmap
# -----------------------------
# OLMo2-7B has many layers; we can keep tik_step=5 like your original 7B default.
tik_step = 5

fig, ax, ax2 = plot_ci_plus_heatmap(
    latent_token_probs,
    entropy,
    "en",
    color="tab:orange",
    tik_step=tik_step,
    do_colorbar=True,
    nums=[0.99, 0.18, 0.025, 0.6],
)

if target_lang != "en":
    plot_ci(ax2, out_token_probs, target_lang, color="tab:blue", do_lines=False)

ax2.set_xlabel("layer")
ax2.set_ylabel("probability")
ax2.set_xlim(0, out_token_probs.shape[1] + 1)
ax2.set_ylim(0, 1)
ax2.legend(loc="upper left")

plot1_path = os.path.join(save_dir, f"olmo2_{input_lang}_{target_lang}_probas_ent{suffix}.pdf")
plt.savefig(plot1_path, dpi=300, bbox_inches="tight")
print(f"Saved plot to: {plot1_path}")


# -----------------------------
# Plot 2: energy
# -----------------------------
fig2, axE = plt.subplots(figsize=(5, 3))
plot_ci(axE, energy, "energy", color="tab:green", do_lines=True, tik_step=tik_step)
axE.set_xlabel("layer")
axE.set_ylabel("energy")
axE.set_xlim(0, out_token_probs.shape[1] + 1)

plot2_path = os.path.join(save_dir, f"olmo2_{input_lang}_{target_lang}_energy{suffix}.pdf")
plt.savefig(plot2_path, dpi=300, bbox_inches="tight")
print(f"Saved plot to: {plot2_path}")


# -----------------------------
# Save latents tensor
# -----------------------------
latents_path = os.path.join(save_dir, f"olmo2_{input_lang}_{target_lang}_latents{suffix}.pt")
torch.save(latents, latents_path)
print(f"Saved latents to: {latents_path}")
