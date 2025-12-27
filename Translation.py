#!/usr/bin/env python3
"""
Translation.py (OLMo-2 version) with argparse.

Runs the "latent language" translation analysis using:
- HF hub model (default): allenai/OLMo-2-1124-7B
- or a local HF-converted checkpoint (continual pretrain) via --model_id_or_path /path/to/ckpt

Example:
  python Translation.py \
    --input_lang fr --target_lang zh \
    --model_id_or_path allenai/OLMo-2-1124-7B \
    --prefix /scratch/e1583535/llm-latent-language/data/langs/ \
    --out_dir /scratch/e1583535/llm-latent-language/visuals \
    --k_fewshot 5 --load_in_8bit

Local continual checkpoint:
  python Translation.py \
    --input_lang fr --target_lang zh \
    --model_id_or_path /scratch/e1583535/llm/nus-olmo/para-only-7B-34B-checkpoints/step8290-unsharded-hf \
    --prefix /scratch/e1583535/llm-latent-language/data/langs/ \
    --out_dir /scratch/e1583535/llm-latent-language/visuals \
    --k_fewshot 5 --load_in_8bit
"""

import argparse
import os
from typing import Optional, List, Dict, Any

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
# Helper: OLMo-2 wrapper
# -----------------------------
class Olmo2Helper:
    """
    Minimal helper to:
      - load tokenizer/model from HF hub or local path
      - return hidden states for all layers
      - expose "unemb" = final_norm -> lm_head
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

        # Find final norm
        base = getattr(self.model, "model", None) or getattr(self.model, "base_model", None) or self.model
        if hasattr(base, "norm"):
            self.final_norm = base.norm
        else:
            for cand in ["final_layer_norm", "final_layernorm", "ln_f"]:
                if hasattr(base, cand):
                    self.final_norm = getattr(base, cand)
                    break
            else:
                raise AttributeError(
                    "Could not find a final norm module on this checkpoint. "
                    "Tried: model.norm, final_layer_norm, final_layernorm, ln_f."
                )

        self.unemb = nn.Sequential(self.final_norm, self.lm_head)

    @torch.no_grad()
    def latents_all_layers(self, prompt: str) -> torch.Tensor:
        """
        Returns: latents of shape [L, B, T, H] for L transformer layers.
        Drops embedding output state.
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
    ids = tokenizer.encode(token_str, add_special_tokens=False)
    return list(set(ids))


def compute_entropy(probas: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    return (-probas * torch.log2(probas.clamp_min(eps))).sum(dim=-1)


lang2name = {"fr": "Français", "de": "Deutsch", "ru": "Русский", "en": "English", "zh": "中文"}


def build_fewshot_prompt(
    df: pd.DataFrame,
    ind: int,
    k: int,
    lang1: str,
    lang2: str,
    tokenizer,
    seed: int,
) -> Optional[Dict[str, Any]]:
    df = df.reset_index(drop=True)
    if ind < 0 or ind >= len(df):
        return None

    temp = df[df.index != ind]
    sample_rows = pd.concat([temp.sample(k - 1, random_state=seed), df[df.index == ind]], axis=0)

    prompt = ""
    out_token_str = None
    in_token_str = None
    latent_token_str = None

    for idx, (_, row) in enumerate(sample_rows.iterrows()):
        if idx < k - 1:
            prompt += f'{lang2name[lang1]}: "{row[lang1]}" - {lang2name[lang2]}: "{row[lang2]}"\n'
        else:
            prompt += f'{lang2name[lang1]}: "{row[lang1]}" - {lang2name[lang2]}: "'
            in_token_str = row[lang1]
            out_token_str = row[lang2]
            latent_token_str = row["en"]  # latent language fixed to English

    if out_token_str is None or in_token_str is None or latent_token_str is None:
        return None

    out_token_id = process_tokens(out_token_str, tokenizer)
    latent_token_id = process_tokens(latent_token_str, tokenizer)

    if len(out_token_id) == 0 or len(latent_token_id) == 0:
        return None

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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="OLMo-2 Translation latent-language analysis")

    # Languages / data
    ap.add_argument("--input_lang", default="fr", help="Input language code (e.g., fr)")
    ap.add_argument("--target_lang", default="zh", help="Target language code (e.g., zh)")
    ap.add_argument("--k_fewshot", type=int, default=5, help="Few-shot examples k (k-1 demonstrations + 1 query)")
    ap.add_argument("--prefix", default="llm-latent-language/data/langs/",
                    help="Prefix directory containing <lang>/clean.csv")
    ap.add_argument("--out_dir", default="./visuals",
                    help="Output directory to save CSV/PDF/PT results")

    # Model loading
    ap.add_argument("--model_id_or_path", default="allenai/OLMo-2-1124-7B",
                    help="HF model id (hub) or local HF checkpoint directory")
    ap.add_argument("--revision", default=None, help="HF revision (tag/commit/branch) for hub models")
    ap.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit (requires bitsandbytes)")
    ap.add_argument("--no_load_in_8bit", dest="load_in_8bit", action="store_false",
                    help="Disable 8-bit loading (use fp16/bf16)")
    ap.set_defaults(load_in_8bit=True)

    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                    help="dtype when not using 8-bit")

    # Filtering
    ap.add_argument("--single_token_only", action="store_true", help="Keep only target strings that are 1 token")
    ap.add_argument("--multi_token_only", action="store_true", help="Keep only target strings with >1 token")

    # Misc
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--max_items", type=int, default=None,
                    help="Optional cap on dataset size for quick debugging (e.g., 50)")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.single_token_only and args.multi_token_only:
        raise ValueError("single_token_only and multi_token_only cannot both be True.")

    input_lang = args.input_lang
    target_lang = args.target_lang

    # Load data
    df_en_in = pd.read_csv(f"{args.prefix}{input_lang}/clean.csv").reindex()
    df_en_tgt = pd.read_csv(f"{args.prefix}{target_lang}/clean.csv").reindex()

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

    # Remove rows where English appears inside target string
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

    # dtype
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Load model
    olmo = Olmo2Helper(
        model_id_or_path=args.model_id_or_path,
        revision=args.revision,
        load_in_8bit=args.load_in_8bit,
        dtype=dtype,
    )
    tokenizer = olmo.tokenizer
    unemb = olmo.unemb

    print(f"Loaded model from: {args.model_id_or_path}")
    if args.revision is not None:
        print(f"  revision: {args.revision}")

    # Build U_normalized and avgUU for "energy"
    with torch.no_grad():
        U = list(unemb[1].parameters())[0].detach().cpu().float()  # [V, H]
        weights = list(unemb[0].parameters())[0].detach().cpu().float()  # [H]
        U_weighted = U * weights.unsqueeze(0)
        U_normalized = U_weighted / ((U_weighted**2).sum(dim=1, keepdim=True)).sqrt()

        v = U.shape[0]
        avgUU = (((U_normalized.T @ U_normalized) ** 2).sum() / (v**2)).sqrt()
        print(f"U {U.shape} | norm weights {weights.shape} | avgUU={avgUU.item():.6f}")

    # Optional token-length filtering
    def num_tokens(s: str) -> int:
        return len(tokenizer.encode(str(s), add_special_tokens=False))

    if args.single_token_only or args.multi_token_only:
        keep_rows = []
        for i, word in enumerate(df_merged[target_lang].tolist()):
            n = num_tokens(word)
            if args.single_token_only and n == 1:
                keep_rows.append(i)
            if args.multi_token_only and n > 1:
                keep_rows.append(i)
        df_merged = df_merged.iloc[keep_rows].reset_index(drop=True)
        print(f"After token-length filtering: {len(df_merged)} rows")

    # Build dataset
    dataset: List[Dict[str, Any]] = []
    for ind in tqdm(range(len(df_merged)), desc="Building dataset"):
        d = build_fewshot_prompt(
            df_merged, ind, k=args.k_fewshot, lang1=input_lang, lang2=target_lang, tokenizer=tokenizer, seed=args.seed
        )
        if d is None:
            continue
        dataset.append(d)
        if args.max_items is not None and len(dataset) >= args.max_items:
            break

    df_dataset = pd.DataFrame(dataset)

    # Output directory (safe name even for absolute path)
    run_name = os.path.basename(os.path.normpath(args.model_id_or_path))
    save_dir = os.path.join(args.out_dir, run_name, "translation")
    os.makedirs(save_dir, exist_ok=True)

    suffix = ""
    if args.single_token_only:
        suffix = "_single_token"
    elif args.multi_token_only:
        suffix = "_multi_token"

    dataset_csv = os.path.join(save_dir, f"olmo2_{input_lang}_{target_lang}_dataset{suffix}.csv")
    df_dataset.to_csv(dataset_csv, index=False)
    print(f"Saved dataset to: {dataset_csv}")
    print(df_dataset.head())

    # Run analysis (per layer)
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

        # store final-position latents [L, H] (batch=0, last token)
        lat_last = latents[:, 0, -1, :].float()                 # [L, H]
        latents_all.append(lat_last.detach().cpu().clone())

        # energy
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

    # Plot 1: probs + entropy heatmap
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

    # Plot 2: energy
    fig2, axE = plt.subplots(figsize=(5, 3))
    plot_ci(axE, energy, "energy", color="tab:green", do_lines=True, tik_step=tik_step)
    axE.set_xlabel("layer")
    axE.set_ylabel("energy")
    axE.set_xlim(0, out_token_probs.shape[1] + 1)

    plot2_path = os.path.join(save_dir, f"olmo2_{input_lang}_{target_lang}_energy{suffix}.pdf")
    plt.savefig(plot2_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {plot2_path}")

    # Save latents tensor
    latents_path = os.path.join(save_dir, f"olmo2_{input_lang}_{target_lang}_latents{suffix}.pt")
    torch.save(latents, latents_path)
    print(f"Saved latents to: {latents_path}")


if __name__ == "__main__":
    main()
