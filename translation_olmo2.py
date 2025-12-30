#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM

# From the original repo
from utils import plot_ci, plot_ci_plus_heatmap


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# OLMo-2 Helper (close to LlamaHelper usage)
# -----------------------------
class Olmo2Helper:
    """
    Minimal wrapper to match notebook usage:
      - .tokenizer
      - .model
      - .latents_all_layers(prompt) -> [L, T, H] (batch squeezed)
    """

    def __init__(
        self,
        dir: str,
        load_in_8bit: bool = True,
        revision: Optional[str] = None,
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
                dtype=dtype,
            )

        self.model.eval()

    @torch.no_grad()
    def latents_all_layers(self, prompt: str) -> torch.Tensor:
        """
        Returns hidden states for every transformer layer:
          shape [L, T, H]
        Drops embedding output to match original layer indexing.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[1:]                 # drop embedding output: tuple len = L
        return torch.stack(hs, dim=0)[:, 0, :, :]  # [L, T, H]


# -----------------------------
# Token utilities (Notebook logic + OLMo-2 fallback)
# -----------------------------
def token_prefixes(token_str: str):
    n = len(token_str)
    return [token_str[:i] for i in range(1, n + 1)]

def add_spaces(tokens):
    return ["▁" + t for t in tokens] + tokens

def capitalizations(tokens):
    return list(set(tokens))

def unicode_prefix_tokid(zh_char="云", tokenizer=None):
    if tokenizer is None:
        return None
    try:
        start = zh_char.encode().__str__()[2:-1].split("\\x")[1]
    except Exception:
        return None
    start_key = f"<0x{start.upper()}>"
    vocab = tokenizer.get_vocab()
    return vocab.get(start_key, None)

def process_tokens(token_str: str, tokenizer, lang: str) -> List[int]:
    """
    Notebook-style vocab matching, PLUS fallback to tokenizer.encode()
    (needed for OLMo-2, especially for zh).
    """
    token_str = str(token_str)

    vocab = tokenizer.get_vocab()
    with_prefixes = token_prefixes(token_str)
    with_spaces = add_spaces(with_prefixes)
    with_caps = capitalizations(with_spaces)

    final_tokens = [vocab[tok] for tok in with_caps if tok in vocab]

    if lang in ["zh", "ru"]:
        tokid = unicode_prefix_tokid(token_str, tokenizer)
        if tokid is not None:
            final_tokens.append(tokid)

    # OLMo-2 fallback (critical for zh)
    if len(final_tokens) == 0:
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        final_tokens = list(set(ids))

    return final_tokens

def compute_entropy(probas: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    return (-probas * torch.log2(probas.clamp_min(eps))).sum(dim=-1)


# -----------------------------
# Few-shot prompt builder (same spirit as notebook)
# -----------------------------
lang2name = {"fr": "Français", "de": "Deutsch", "ru": "Русский", "en": "English", "zh": "中文"}

def build_translation_example(
    df: pd.DataFrame,
    ind: int,
    k: int,
    tokenizer,
    lang1: str,
    lang2: str,
    lang_latent: str = "en",
    seed: int = 42,
) -> Optional[Dict[str, Any]]:
    df = df.reset_index(drop=True)
    if ind < 0 or ind >= len(df):
        return None

    temp = df[df.index != ind]
    # fix random_state for determinism
    sample_rows = pd.concat([temp.sample(k - 1, random_state=seed + ind), df[df.index == ind]], axis=0)

    prompt = ""
    in_token_str = out_token_str = latent_token_str = None

    for idx, (_, row) in enumerate(sample_rows.iterrows()):
        if idx < k - 1:
            prompt += f'{lang2name[lang1]}: "{row[lang1]}" - {lang2name[lang2]}: "{row[lang2]}"\n'
        else:
            prompt += f'{lang2name[lang1]}: "{row[lang1]}" - {lang2name[lang2]}: "'
            in_token_str = row[lang1]
            out_token_str = row[lang2]
            latent_token_str = row[lang_latent]  # English

    if in_token_str is None or out_token_str is None or latent_token_str is None:
        return None

    out_token_id = process_tokens(out_token_str, tokenizer, lang2)
    latent_token_id = process_tokens(latent_token_str, tokenizer, "en")

    if len(out_token_id) == 0 or len(latent_token_id) == 0:
        return None

    # Same filtering as notebook: drop if overlap when target != en
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


def num_tokens(s: str, tokenizer) -> int:
    return len(tokenizer.encode(str(s), add_special_tokens=False))


def main():
    parser = argparse.ArgumentParser(description="Translation (OLMo-2) - close to Translation.ipynb")

    parser.add_argument("--source_lang", type=str, required=True)
    parser.add_argument("--target_lang", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, help="HF hub id or local HF checkpoint path")
    parser.add_argument("--revision", type=str, default=None)

    parser.add_argument("--data_prefix", type=str, required=True, help="Path containing <lang>/clean.csv")
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--k_fewshot", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--no_load_in_8bit", dest="load_in_8bit", action="store_false")
    parser.set_defaults(load_in_8bit=True)

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    parser.add_argument("--img_ext", type=str, default="png", choices=["png", "jpg", "jpeg", "svg"])
    parser.add_argument("--dpi", type=int, default=300)

    parser.add_argument("--single_token_only", action="store_true")
    parser.add_argument("--multi_token_only", action="store_true")

    parser.add_argument("--max_examples", type=int, default=0, help="Debug: limit dataset size (0 = no limit)")

    args = parser.parse_args()
    set_seed(args.seed)

    if args.single_token_only and args.multi_token_only:
        raise ValueError("--single_token_only and --multi_token_only cannot both be True")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    input_lang = args.source_lang
    target_lang = args.target_lang

    # -----------------------------
    # Load model + tokenizer
    # -----------------------------
    olmo = Olmo2Helper(dir=args.model, load_in_8bit=args.load_in_8bit, revision=args.revision, dtype=dtype)
    tokenizer = olmo.tokenizer
    model = olmo.model

    # Build unemb = final_norm -> lm_head (like notebook)
    base = getattr(model, "model", None) or getattr(model, "base_model", None) or model
    if hasattr(base, "norm"):
        final_norm = base.norm
    else:
        for cand in ["final_layer_norm", "final_layernorm", "ln_f"]:
            if hasattr(base, cand):
                final_norm = getattr(base, cand)
                break
        else:
            raise AttributeError("Could not find final norm module (model.norm / ln_f / etc.)")

    lm_head = model.get_output_embeddings()
    unemb = nn.Sequential(final_norm, lm_head)

    # -----------------------------
    # Energy prep (same math as notebook, but with correct shapes)
    # -----------------------------
    with torch.no_grad():
        U = list(unemb[1].parameters())[0].detach().cpu().float()        # [V, H]
        weights = list(unemb[0].parameters())[0].detach().cpu().float()  # [H]
        U_weighted = U * weights.unsqueeze(0)
        U_normalized = U_weighted / ((U_weighted**2).sum(dim=1, keepdim=True)).sqrt()
        v = U.shape[0]
        avgUU = (((U_normalized.T @ U_normalized) ** 2).sum() / (v**2)).sqrt()
        print(f"U {U.shape} | weights {weights.shape} | avgUU={avgUU.item():.6f}")

    # -----------------------------
    # Load data (same join/rename as notebook)
    # -----------------------------
    df_src = pd.read_csv(os.path.join(args.data_prefix, input_lang, "clean.csv")).reset_index(drop=True)
    df_tgt = pd.read_csv(os.path.join(args.data_prefix, target_lang, "clean.csv")).reset_index(drop=True)

    if input_lang == target_lang:
        df_merged = df_tgt.copy()
        df_merged.rename(
            columns={
                "word_original": "en",
                "word_translation": target_lang if target_lang != "en" else "en_tgt",
            },
            inplace=True,
        )
    else:
        df_merged = df_tgt.merge(df_src, on=["word_original"], suffixes=(f"_{target_lang}", f"_{input_lang}"))
        df_merged.rename(
            columns={
                "word_original": "en",
                f"word_translation_{target_lang}": target_lang if target_lang != "en" else "en_tgt",
                f"word_translation_{input_lang}": input_lang if input_lang != "en" else "en_in",
            },
            inplace=True,
        )

    # Delete rows where English appears inside target translation (notebook)
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

    # Optional token-length filtering (OLMo-2-safe)
    if args.single_token_only or args.multi_token_only:
        keep = []
        for i, w in enumerate(df_merged[target_lang].tolist()):
            n = num_tokens(w, tokenizer)
            if args.single_token_only and n == 1:
                keep.append(i)
            if args.multi_token_only and n > 1:
                keep.append(i)
        df_merged = df_merged.iloc[keep].reset_index(drop=True)
        print(f"After token-length filtering: {len(df_merged)} rows")

    # -----------------------------
    # Build dataset
    # -----------------------------
    dataset: List[Dict[str, Any]] = []
    for ind in tqdm(range(len(df_merged)), desc="Building dataset"):
        d = build_translation_example(
            df=df_merged,
            ind=ind,
            k=args.k_fewshot,
            tokenizer=tokenizer,
            lang1=input_lang,
            lang2=target_lang,
            lang_latent="en",
            seed=args.seed,
        )
        if d is None:
            continue
        dataset.append(d)
        if args.max_examples > 0 and len(dataset) >= args.max_examples:
            break

    print(f"final dataset length: {len(dataset)}")
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty after filtering. Try increasing data, reducing filtering, or adjusting token logic.")

    # -----------------------------
    # Save dataset CSV
    # -----------------------------
    run_name = os.path.basename(os.path.normpath(args.model))
    save_dir = os.path.join(args.out_dir, run_name, "translation")
    os.makedirs(save_dir, exist_ok=True)

    suffix = ""
    if args.single_token_only:
        suffix = "_single_token"
    elif args.multi_token_only:
        suffix = "_multi_token"

    df_out = pd.DataFrame(dataset)
    dataset_path = os.path.join(save_dir, f"{input_lang}_{target_lang}_dataset{suffix}.csv")
    df_out.to_csv(dataset_path, index=False)
    print("Saved dataset:", dataset_path)

    # -----------------------------
    # Run analysis (same as notebook)
    # -----------------------------
    latent_token_probs = []
    out_token_probs = []
    entropy = []
    energy = []
    latents_all = []

    for d in tqdm(dataset, desc="Running model"):
        latents = olmo.latents_all_layers(d["prompt"])          # [L,T,H]
        logits = unemb(latents)                                  # [L,T,V]
        last = logits[:, -1, :].float().softmax(dim=-1)          # [L,V]

        latent_ids = torch.tensor(d["latent_token_id"], dtype=torch.long, device=last.device)
        out_ids = torch.tensor(d["out_token_id"], dtype=torch.long, device=last.device)

        latent_token_probs.append(last[:, latent_ids].sum(dim=-1).detach().cpu())  # [L]
        out_token_probs.append(last[:, out_ids].sum(dim=-1).detach().cpu())        # [L]
        entropy.append(compute_entropy(last).detach().cpu())                       # [L]

        # latents at final position
        lat_last = latents[:, -1, :].float()                                        # [L,H]
        latents_all.append(lat_last.detach().cpu().clone())

        # energy (FIXED shape): (V,H) @ (H,L) -> (V,L) -> reduce over V -> [L]
        lat_norm = lat_last
        lat_norm = lat_norm / (((lat_norm**2).mean(dim=-1, keepdim=True)).sqrt() + 1e-12)
        lat_norm = lat_norm / (lat_norm.norm(dim=-1, keepdim=True) + 1e-12)

        proj = U_normalized.to(lat_norm.device) @ lat_norm.T   # [V,L]
        norm_val = (proj.pow(2).mean(dim=0)).sqrt()            # [L]
        energy.append((norm_val / avgUU.to(lat_norm.device)).detach().cpu())

    latent_token_probs = torch.stack(latent_token_probs)  # [N,L]
    out_token_probs = torch.stack(out_token_probs)        # [N,L]
    entropy = torch.stack(entropy)                        # [N,L]
    energy = torch.stack(energy)                          # [N,L]
    latents_tensor = torch.stack(latents_all)             # [N,L,H]

    # -----------------------------
    # Plot 1: probs + entropy heatmap
    # -----------------------------
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

    plot1 = os.path.join(save_dir, f"{input_lang}_{target_lang}_probas_ent{suffix}.{args.img_ext}")
    plt.savefig(plot1, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", plot1)

    # -----------------------------
    # Plot 2: energy
    # -----------------------------
    fig2, axE = plt.subplots(figsize=(5, 3))
    plot_ci(axE, energy, "energy", color="tab:green", do_lines=True, tik_step=tik_step)
    axE.set_xlabel("layer")
    axE.set_ylabel("energy")
    axE.set_xlim(0, out_token_probs.shape[1] + 1)

    plot2 = os.path.join(save_dir, f"{input_lang}_{target_lang}_energy{suffix}.{args.img_ext}")
    plt.savefig(plot2, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig2)
    print("Saved:", plot2)

    # -----------------------------
    # Save latents tensor
    # -----------------------------
    lat_path = os.path.join(save_dir, f"{input_lang}_{target_lang}_latents{suffix}.pt")
    torch.save(latents_tensor, lat_path)
    print("Saved:", lat_path)


if __name__ == "__main__":
    main()
