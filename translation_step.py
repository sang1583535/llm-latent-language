#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt
from untils import plot_ci, plot_ci_plus_heatmap

from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# OLMo-2 Helper (same as translation_olmo2.py)
# -----------------------------
class Olmo2Helper:
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
                dir, revision=revision, device_map="auto", load_in_8bit=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                dir, revision=revision, device_map="auto", torch_dtype=dtype
            )

        self.model.eval()

    @torch.no_grad()
    def latents_all_layers(self, prompt: str) -> torch.Tensor:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[1:]                 # drop embedding output
        return torch.stack(hs, dim=0)[:, 0, :, :]  # [L,T,H]


# -----------------------------
# Token utilities (same as translation_olmo2.py)
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

    # OLMo-2 fallback (important for zh)
    if len(final_tokens) == 0:
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        final_tokens = list(set(ids))

    return final_tokens

lang2name = {"fr": "Français", "de": "Deutsch", "ru": "Русский", "en": "English", "zh": "中文"}

def num_tokens(s: str, tokenizer) -> int:
    return len(tokenizer.encode(str(s), add_special_tokens=False))


# -----------------------------
# Few-shot prompt builder (same spirit as translation_olmo2.py)
# -----------------------------
def build_translation_example(
    df: pd.DataFrame,
    ind: int,
    k: int,
    tokenizer,
    lang1: str,
    lang2: str,
    seed: int,
    lang_latent: str = "en",
) -> Optional[Dict[str, Any]]:
    df = df.reset_index(drop=True)
    if ind < 0 or ind >= len(df):
        return None

    temp = df[df.index != ind]
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

    # Same filtering as notebook: skip overlap when target != en
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
# Main
# -----------------------------
def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="Translation step-wise analysis (OLMo-2), close to translation_olmo2.py")

    ap.add_argument("--source_lang", type=str, required=True)
    ap.add_argument("--target_lang", type=str, required=True)

    ap.add_argument("--data_prefix", type=str, required=True, help="Path containing <lang>/clean.csv")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument(
        "--checkpoints", type=str, required=True,
        help="Comma-separated list of HF checkpoint paths or hub ids."
    )
    ap.add_argument(
        "--steps", type=str, required=True,
        help="Comma-separated list of step labels (same length as checkpoints)."
    )

    ap.add_argument("--revision", type=str, default=None)
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--no_load_in_8bit", dest="load_in_8bit", action="store_false")
    ap.set_defaults(load_in_8bit=True)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    ap.add_argument("--k_fewshot", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--layers", type=str, default="10,20,30", help="Comma-separated layer indices to plot (x=step).")

    ap.add_argument("--single_token_only", action="store_true")
    ap.add_argument("--multi_token_only", action="store_true")

    ap.add_argument("--max_examples", type=int, default=0, help="Debug: limit dataset size (0 = no limit)")

    ap.add_argument("--img_ext", type=str, default="png", choices=["png", "jpg", "jpeg", "svg"])
    ap.add_argument("--dpi", type=int, default=300)

    args = ap.parse_args()
    set_seed(args.seed)

    if args.single_token_only and args.multi_token_only:
        raise ValueError("--single_token_only and --multi_token_only cannot both be True")

    ckpts = [x.strip() for x in args.checkpoints.split(",") if x.strip()]
    steps = [x.strip() for x in args.steps.split(",") if x.strip()]
    if len(ckpts) != len(steps):
        raise ValueError("--checkpoints and --steps must have the same length")

    layer_list = parse_int_list(args.layers)
    if not layer_list:
        raise ValueError("--layers must be non-empty, e.g. 10,20,30")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    input_lang = args.source_lang
    target_lang = args.target_lang

    # -----------------------------
    # Load data and build merged table (same as translation_olmo2.py)
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
    # Tokenizer from first checkpoint, then token-length filtering + dataset once
    # -----------------------------
    helper0 = Olmo2Helper(dir=ckpts[0], load_in_8bit=args.load_in_8bit, revision=args.revision, dtype=dtype)
    tokenizer = helper0.tokenizer

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

    dataset: List[Dict[str, Any]] = []
    for ind in tqdm(range(len(df_merged)), desc="Building dataset"):
        d = build_translation_example(
            df=df_merged,
            ind=ind,
            k=args.k_fewshot,
            tokenizer=tokenizer,
            lang1=input_lang,
            lang2=target_lang,
            seed=args.seed,
        )
        if d is None:
            continue
        dataset.append(d)
        if args.max_examples > 0 and len(dataset) >= args.max_examples:
            break

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty after filtering. Try relaxing filters or adjusting token logic.")

    print(f"Dataset size: {len(dataset)} prompts")

    # Output dir
    save_dir = os.path.join(args.out_dir, "translation_steps", f"{input_lang}_{target_lang}")
    os.makedirs(save_dir, exist_ok=True)

    # Save dataset once
    pd.DataFrame(dataset).to_csv(os.path.join(save_dir, "dataset.csv"), index=False)

    # results: per layer -> list over steps
    en_means = {layer: [] for layer in layer_list}
    en_stds = {layer: [] for layer in layer_list}
    tgt_means = {layer: [] for layer in layer_list}
    tgt_stds = {layer: [] for layer in layer_list}

    # -----------------------------
    # Loop checkpoints: compute per-layer per-prompt probs then aggregate
    # -----------------------------
    per_layer_en_by_step = {layer: [] for layer in layer_list}
    per_layer_tgt_by_step = {layer: [] for layer in layer_list}

    for ckpt, step in zip(ckpts, steps):
        print(f"\n=== Step {step}: {ckpt} ===")

        olmo = Olmo2Helper(dir=ckpt, load_in_8bit=args.load_in_8bit, revision=args.revision, dtype=dtype)
        model = olmo.model

        # build unemb (same as translation_olmo2.py)
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

        # collect per-prompt values for the layers we care about
        step_vals_en = {layer: [] for layer in layer_list}
        step_vals_tgt = {layer: [] for layer in layer_list}

        for d in tqdm(dataset, desc=f"Running step={step}"):
            latents = olmo.latents_all_layers(d["prompt"])          # [L,T,H]
            logits = unemb(latents)                                  # [L,T,V]
            last = logits[:, -1, :].float().softmax(dim=-1)          # [L,V]

            latent_ids = torch.tensor(d["latent_token_id"], dtype=torch.long, device=last.device)
            out_ids = torch.tensor(d["out_token_id"], dtype=torch.long, device=last.device)

            p_en_all = last[:, latent_ids].sum(dim=-1)               # [L]
            p_tg_all = last[:, out_ids].sum(dim=-1)                  # [L]

            for layer in layer_list:
                if 0 <= layer < p_en_all.shape[0]:
                    step_vals_en[layer].append(float(p_en_all[layer].detach().cpu()))
                    step_vals_tgt[layer].append(float(p_tg_all[layer].detach().cpu()))


        # aggregate with population variance + clamp (matches original style)
        for layer in layer_list:
            per_layer_en_by_step[layer].append(step_vals_en[layer])
            per_layer_tgt_by_step[layer].append(step_vals_tgt[layer])

    # -----------------------------
    # Plot per layer: x=steps, y=mean prob, CI = 1.96*std/sqrt(N)
    # -----------------------------
    x = np.arange(len(steps))
    n = len(dataset)

    for layer in layer_list:
        en_mat = np.array(per_layer_en_by_step[layer], dtype=np.float32).T
        tg_mat = np.array(per_layer_tgt_by_step[layer], dtype=np.float32).T

        fig, ax = plt.subplots(figsize=(6, 3.5))
        plot_ci(ax, en_mat, "en", tik_step=1, do_lines=False)
        plot_ci(ax, tg_mat, target_lang, tik_step=1, do_lines=False)

        ax.set_title(f"Translation | {input_lang}→{target_lang} | layer {layer}")
        ax.set_xlabel("training step")
        ax.set_ylabel("probability")
        ax.set_xticks(x)
        ax.set_xticklabels(steps, rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left")
        fig.tight_layout()

        out_path = os.path.join(save_dir, f"layer_{layer}.{args.img_ext}")
        plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out_path)

    # Save raw numbers
    summary = {
        "steps": steps,
        "layers": layer_list,
        "n_prompts": len(dataset),
        "en_mean": {str(l): en_means[l] for l in layer_list},
        "en_std": {str(l): en_stds[l] for l in layer_list},
        f"{target_lang}_mean": {str(l): tgt_means[l] for l in layer_list},
        f"{target_lang}_std": {str(l): tgt_stds[l] for l in layer_list},
    }
    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Saved:", os.path.join(save_dir, "summary.json"))


if __name__ == "__main__":
    main()
