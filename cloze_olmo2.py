import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

# From original repo
from utils import plot_ci, plot_ci_plus_heatmap


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# OLMo-2 Helper (replacement for LlamaHelper)
# -----------------------------
class Olmo2Helper:
    """
    Minimal wrapper to match the original notebook usage:
      - .tokenizer
      - .model
      - .latents_all_layers(prompt) -> [L, T, H] (batch squeezed)
    """

    def __init__(
        self,
        dir: str,
        load_in_8bit: bool = True,
        revision: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.dir = dir
        self.tokenizer = AutoTokenizer.from_pretrained(dir, revision=revision, use_fast=True)

        if load_in_8bit:
            self.model = AutoModelForCausalLM.from_pretrained(
                dir,
                revision=revision,
                device_map="auto",
                load_in_8bit=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                dir,
                revision=revision,
                device_map="auto",
                torch_dtype=dtype,
            )

        self.model.eval()

    @torch.no_grad()
    def latents_all_layers(self, prompt: str) -> torch.Tensor:
        """
        Returns hidden states for every transformer layer:
          shape [L, T, H]
        (drops embedding output to match original layer indexing)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # (emb, layer1, ..., layerN)
        hs = hs[1:]             # drop embedding output

        # stack -> [L, B, T, H], then squeeze batch -> [L, T, H]
        return torch.stack(hs, dim=0)[:, 0, :, :]


# -----------------------------
# Token utilities (copied from notebook)
# -----------------------------
def token_prefixes(token_str: str):
    n = len(token_str)
    tokens = [token_str[:i] for i in range(1, n + 1)]
    return tokens


def add_spaces(tokens):
    # return ["▁" + t for t in tokens] + tokens
    return ['Ġ' + t for t in tokens] +  ['▁' + t for t in tokens] + tokens


def capitalizations(tokens):
    return list(set(tokens))


def unicode_prefix_tokid(zh_char="云", tokenizer=None):
    """
    Notebook behavior:
    - encodes the first UTF-8 byte of zh_char into <0x..> token
    - if that token exists in vocab, return its token id
    """
    if tokenizer is None:
        return None

    try:
        start = zh_char.encode().__str__()[2:-1].split("\\x")[1]
    except Exception:
        return None

    unicode_format = "<0x%s>"
    start_key = unicode_format % start.upper()
    vocab = tokenizer.get_vocab()
    if start_key in vocab:
        return vocab[start_key]
    return None


def process_tokens(token_str: str, tokenizer, lang: str):
    """
    Notebook behavior:
      - enumerate string prefixes
      - add spaces (▁prefix) variants
      - dedupe
      - keep only those that exist in vocab (string lookup)
      - for zh/ru also try unicode_prefix_tokid
    """
    token_str = str(token_str)
    with_prefixes = token_prefixes(token_str)
    with_spaces = add_spaces(with_prefixes)
    with_capitalizations = capitalizations(with_spaces)
    final_tokens = []
    vocab = tokenizer.get_vocab()

    for tok in with_capitalizations:
        if tok in vocab:
            final_tokens.append(vocab[tok])

    if lang in ["zh", "ru"]:
        tokid = unicode_prefix_tokid(token_str, tokenizer)
        if tokid is not None:
            final_tokens.append(tokid)

    # --- OLMo-2-safe fallback (crucial for zh) ---
    if len(final_tokens) == 0:
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        final_tokens = list(set(ids))

    return final_tokens


def compute_entropy(probas: torch.Tensor):
    eps = 1e-12
    return (-probas * torch.log2(probas.clamp_min(eps))).sum(dim=-1)


lang2name = {"fr": "Français", "de": "Deutsch", "ru": "Русский", "en": "English", "zh": "中文"}


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_lang", type=str, default="fr")
    ap.add_argument("--model_size", type=str, default="7b", choices=["7b", "13b", "70b"])  # kept for tik_step
    ap.add_argument("--model", type=str, required=True, help="HF hub id or local HF checkpoint dir for OLMo-2")
    ap.add_argument("--revision", type=str, default=None)

    ap.add_argument("--data_prefix", type=str, default="./data/langs/", help="Path containing <lang>/clean.csv")
    ap.add_argument("--out_dir", type=str, default="./visuals")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--n_skip", type=int, default=2, help="Number of demo cloze lines")
    ap.add_argument("--key", type=str, default="blank_prompt_translation_masked")
    ap.add_argument("--max_examples", type=int, default=0, help="0 = use all, else truncate dataset")

    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--no_load_in_8bit", dest="load_in_8bit", action="store_false")
    ap.set_defaults(load_in_8bit=True)

    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--img_ext", type=str, default="png", choices=["png", "jpg", "jpeg", "svg"])
    ap.add_argument("--dpi", type=int, default=300)

    args = ap.parse_args()
    set_seed(args.seed)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # -----------------------------
    # Load data
    # -----------------------------
    prefix = args.data_prefix
    df_en_de = pd.read_csv(f"{prefix}{args.target_lang}/clean.csv")

    # -----------------------------
    # Load model (OLMo-2)
    # -----------------------------
    llama = Olmo2Helper(dir=args.model, load_in_8bit=args.load_in_8bit, revision=args.revision, dtype=dtype)

    tokenizer = llama.tokenizer
    model = llama.model

    # -----------------------------
    # Build unemb and energy helpers
    # -----------------------------
    base = getattr(model, "model", None) or getattr(model, "base_model", None) or model
    if hasattr(base, "norm"):
        final_norm = base.norm
    else:
        # fallback attempts
        for cand in ["final_layer_norm", "final_layernorm", "ln_f"]:
            if hasattr(base, cand):
                final_norm = getattr(base, cand)
                break
        else:
            raise AttributeError("Could not find final norm module on this checkpoint (model.norm / ln_f / etc.)")

    lm_head = model.get_output_embeddings()
    unemb = nn.Sequential(final_norm, lm_head)

    print(unemb)

    U = list(unemb[1].parameters())[0].detach().cpu().float()         # [V,H]
    weights = list(unemb[0].parameters())[0].detach().cpu().float()   # [H]
    print(f"U {U.shape} weights {weights.unsqueeze(0).shape}")

    U_weighted = U.clone()
    U_weighted *= weights.unsqueeze(0)
    U_normalized = U_weighted / ((U_weighted**2).sum(dim=1, keepdim=True)) ** 0.5
    v = U.shape[0]
    avgUU = (((U_normalized.T @ U_normalized) ** 2).sum() / v**2) ** 0.5
    print("avgUU:", float(avgUU))

    # -----------------------------
    # Gap texts (same logic as notebook)
    # -----------------------------
    key = args.key
    dataset_gap = []
    n_skip = args.n_skip

    for idx, (idx_df, row) in tqdm(enumerate(df_en_de.iterrows()), total=len(df_en_de), desc="Building dataset_gap"):
        prompt_template = ""
        indices = set(list(range(len(df_en_de)))) - set([idx])
        idx_examples = np.random.choice(list(indices), n_skip, replace=False)

        # add demonstration lines
        for ex_i in idx_examples:
            prompt_template += f"{df_en_de[key][ex_i]}\n"

        out_token_str = row["word_translation"]
        latent_token_str = row["word_original"]

        out_token_id = process_tokens(out_token_str, tokenizer, args.target_lang)
        latent_token_id = process_tokens(latent_token_str, tokenizer, "en")

        intersection = set(out_token_id).intersection(set(latent_token_id))
        if len(out_token_id) == 0 or len(latent_token_id) == 0:
            continue
        if args.target_lang != "en" and len(intersection) > 0:
            continue

        if args.target_lang == "zh":
            prompt = row[key].split("：")[0]+": \""
        else:
            prompt = row[key].split(":")[0]+": \""

        dataset_gap.append(
            {
                "prompt": prompt_template + prompt,
                "out_token_id": out_token_id,
                "out_token_str": out_token_str,
                "latent_token_id": latent_token_id,
                "latent_token_str": latent_token_str,
            }
        )

        if args.max_examples > 0 and len(dataset_gap) >= args.max_examples:
            break

    print("len(dataset_gap) =", len(dataset_gap))
    if len(dataset_gap) == 0:
        raise RuntimeError("Dataset is empty after filtering; try different target_lang or relax filtering.")

    df_gap = pd.DataFrame(dataset_gap)
    print(df_gap["prompt"].iloc[0])

    # -----------------------------
    # Save dataset (match notebook structure)
    # -----------------------------
    run_name = os.path.basename(os.path.normpath(args.model))
    save_dir = os.path.join(args.out_dir, run_name, "cloze")
    os.makedirs(save_dir, exist_ok=True)

    df_gap.to_csv(os.path.join(save_dir, f"{args.target_lang}_dataset.csv"), index=False)

    # -----------------------------
    # Forward pass loop (same as notebook)
    # -----------------------------
    latent_token_probs = []
    out_token_probs = []
    entropy = []
    energy = []
    latents_all = []

    for d in tqdm(dataset_gap, desc="Running model"):
        prompt = d["prompt"]
        latents = llama.latents_all_layers(prompt)     # [L,T,H]
        logits = unemb(latents)                        # [L,T,V]
        last = logits[:, -1, :].float().softmax(dim=-1).detach().cpu()  # [L,V]

        latent_token_probs += [last[:, torch.tensor(d["latent_token_id"])].sum(dim=-1)]
        out_token_probs += [last[:, torch.tensor(d["out_token_id"])].sum(dim=-1)]
        entropy += [compute_entropy(last)]

        latents_all += [latents[:, -1, :].float().detach().cpu().clone()]  # [L,H]

        # energy (same math, but correct dims)
        latents_normalized = latents[:, -1, :].float()  # [L,H]
        latents_normalized = latents_normalized / (((latents_normalized**2).mean(dim=-1, keepdim=True)) ** 0.5)
        latents_normalized = latents_normalized / (latents_normalized.norm(dim=-1, keepdim=True) + 1e-12)

        proj = U_normalized.to(latents_normalized.device) @ latents_normalized.T  # [V,L]
        norm = (proj.pow(2).mean(dim=0)).sqrt()                                    # [L]
        energy += [(norm / avgUU.to(latents_normalized.device)).detach().cpu()]

    latent_token_probs = torch.stack(latent_token_probs)  # [N,L]
    out_token_probs = torch.stack(out_token_probs)        # [N,L]
    entropy = torch.stack(entropy)                        # [N,L]
    energy = torch.stack(energy)                          # [N,L]
    latents = torch.stack(latents_all)                    # [N,L,H]

    # -----------------------------
    # Plots (same as notebook, but save as images)
    # -----------------------------
    size2tik = {"7b": 5, "13b": 5, "70b": 10}
    tik_step = size2tik.get(args.model_size, 5)

    fig, ax, ax2 = plot_ci_plus_heatmap(
        latent_token_probs, entropy, "en", color="tab:orange", tik_step=tik_step, do_colorbar=True,
        nums=[0.99, 0.18, 0.025, 0.6],
    )
    if args.target_lang != "en":
        plot_ci(ax2, out_token_probs, args.target_lang, color="tab:blue", do_lines=False)

    ax2.set_xlabel("layer")
    ax2.set_ylabel("probability")
    ax2.set_xlim(0, out_token_probs.shape[1] + 1)
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left")

    out1 = os.path.join(save_dir, f"{args.model_size}_{args.target_lang}_probas_ent.{args.img_ext}")
    plt.savefig(out1, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out1)

    fig2, axE = plt.subplots(figsize=(5, 3))
    plot_ci(axE, energy, "energy", color="tab:green", do_lines=True, tik_step=tik_step)
    axE.set_xlabel("layer")
    axE.set_ylabel("energy")
    axE.set_xlim(0, out_token_probs.shape[1] + 1)

    out2 = os.path.join(save_dir, f"{args.model_size}_{args.target_lang}_energy.{args.img_ext}")
    plt.savefig(out2, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig2)
    print("Saved:", out2)

    # Save latents like notebook
    lat_path = os.path.join(save_dir, f"{args.model_size}_{args.target_lang}_latents.pt")
    torch.save(latents, lat_path)
    print("Saved:", lat_path)

    # Optional: save a tiny summary json
    summary_path = os.path.join(save_dir, f"{args.model_size}_{args.target_lang}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "target_lang": args.target_lang,
                "model": args.model,
                "n_examples": len(dataset_gap),
                "probas_shape": list(out_token_probs.shape),
                "latents_shape": list(latents.shape),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
