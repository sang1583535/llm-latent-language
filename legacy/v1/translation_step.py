#!/usr/bin/env python3
import argparse
import os
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

# your repo utils (already in your project)
from utils import plot_ci


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Minimal OLMo-2 helper
# -----------------------------
class Olmo2Helper:
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
        self.lm_head = self.model.get_output_embeddings()

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
                    "Could not find final norm module. Tried: model.norm, final_layer_norm, final_layernorm, ln_f"
                )

        self.unemb = nn.Sequential(self.final_norm, self.lm_head)

    @torch.no_grad()
    def latents_all_layers(self, prompt: str) -> torch.Tensor:
        """
        Return hidden states stacked: [L, B, T, H]
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # (emb, layer1, ..., layerN)
        hs = hs[1:]             # drop embedding output
        return torch.stack(hs, dim=0)  # [L, B, T, H]


# -----------------------------
# Token utilities (tokenizer-agnostic)
# -----------------------------
def process_tokens(token_str: str, tokenizer) -> List[int]:
    # Use tokenizer encoding (works for OLMo-2 BPE/SentencePiece variants)
    ids = tokenizer.encode(str(token_str), add_special_tokens=False)
    return list(set(ids))


LANG2NAME = {"fr": "Français", "de": "Deutsch", "ru": "Русский", "en": "English", "zh": "中文"}


def build_translation_prompt(
    df: pd.DataFrame,
    ind: int,
    k: int,
    lang_in: str,
    lang_tgt: str,
    seed: int,
    tokenizer,
) -> Optional[Dict[str, Any]]:
    """
    Few-shot translation prompt like original notebook:
      <LangIn>: "<x>" - <LangTgt>: "<y>"  (k-1 examples)
      <LangIn>: "<x_q>" - <LangTgt>: "    (query)

    Also returns token id sets:
      - out_token_id: token ids for correct target word
      - latent_token_id: token ids for English word (df['en'])
    """
    df = df.reset_index(drop=True)
    if ind < 0 or ind >= len(df):
        return None

    rng = np.random.RandomState(seed + ind)  # stable per item

    # pick k-1 other rows
    indices = np.arange(len(df))
    indices = indices[indices != ind]
    if len(indices) < (k - 1):
        return None
    ex_ids = rng.choice(indices, size=(k - 1), replace=False)

    prompt = ""
    for ex_i in ex_ids:
        row = df.iloc[ex_i]
        prompt += f'{LANG2NAME[lang_in]}: "{row[lang_in]}" - {LANG2NAME[lang_tgt]}: "{row[lang_tgt]}"\n'

    rowq = df.iloc[ind]
    prompt += f'{LANG2NAME[lang_in]}: "{rowq[lang_in]}" - {LANG2NAME[lang_tgt]}: "'

    out_token_str = rowq[lang_tgt]
    latent_token_str = rowq["en"]

    out_token_id = process_tokens(out_token_str, tokenizer)
    latent_token_id = process_tokens(latent_token_str, tokenizer)

    if len(out_token_id) == 0 or len(latent_token_id) == 0:
        return None

    # match original filtering: avoid ambiguous overlap when target != en
    if lang_tgt != "en" and len(set(out_token_id).intersection(set(latent_token_id))) > 0:
        return None

    return {
        "prompt": prompt,
        "out_token_id": out_token_id,
        "latent_token_id": latent_token_id,
        "out_token_str": out_token_str,
        "latent_token_str": latent_token_str,
        "in_token_str": rowq[lang_in],
    }


def gaussian_ci(mean: np.ndarray, std: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    # 95% Gaussian CI (same type described for Fig 2) :contentReference[oaicite:2]{index=2}
    if n <= 1:
        return mean, mean
    half = 1.96 * (std / np.sqrt(n))
    return mean - half, mean + half


def parse_int_list(s: str) -> List[int]:
    if s.strip() == "":
        return []
    return [int(x.strip()) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_lang", type=str, default="fr")
    ap.add_argument("--target_lang", type=str, default="zh")
    ap.add_argument("--k_fewshot", type=int, default=5)
    ap.add_argument("--data_prefix", type=str, required=True, help="Path containing <lang>/clean.csv")
    ap.add_argument("--out_dir", type=str, required=True)

    # checkpoints
    ap.add_argument("--checkpoints", type=str, required=True,
                    help="Comma-separated list of checkpoint paths (HF format) OR a single HF hub id.")
    ap.add_argument("--steps", type=str, required=True,
                    help="Comma-separated list of step labels (same length as checkpoints), e.g. 477,4770,8290")

    # model loading
    ap.add_argument("--revision", type=str, default=None)
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--no_load_in_8bit", dest="load_in_8bit", action="store_false")
    ap.set_defaults(load_in_8bit=True)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    # plotting
    ap.add_argument("--layers", type=str, default="10,20,30",
                    help="Comma-separated layer indices to plot over steps.")
    ap.add_argument("--max_examples", type=int, default=0,
                    help="If >0, limit dataset size for faster debugging.")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    set_seed(args.seed)

    ckpts = [x.strip() for x in args.checkpoints.split(",") if x.strip()]
    steps = [x.strip() for x in args.steps.split(",") if x.strip()]
    if len(ckpts) != len(steps):
        raise ValueError(f"--checkpoints and --steps must have same length. Got {len(ckpts)} vs {len(steps)}")

    layer_list = parse_int_list(args.layers)
    if len(layer_list) == 0:
        raise ValueError("--layers must be a non-empty comma-separated list, e.g. 10,20,30")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # --- Build merged translation dataframe (en, input_lang, target_lang)
    df_en_in = pd.read_csv(os.path.join(args.data_prefix, args.input_lang, "clean.csv")).reindex()
    df_en_tgt = pd.read_csv(os.path.join(args.data_prefix, args.target_lang, "clean.csv")).reindex()

    if args.input_lang == args.target_lang:
        df_merged = df_en_tgt.copy()
        df_merged.rename(columns={"word_original": "en", "word_translation": args.target_lang}, inplace=True)
    else:
        df_merged = df_en_tgt.merge(df_en_in, on=["word_original"], suffixes=(f"_{args.target_lang}", f"_{args.input_lang}"))
        df_merged.rename(
            columns={
                "word_original": "en",
                f"word_translation_{args.target_lang}": args.target_lang,
                f"word_translation_{args.input_lang}": args.input_lang,
            },
            inplace=True,
        )

    # drop rows where English is contained in target (same as original)
    if args.target_lang != "en":
        drop_idx = []
        for i, row in df_merged.iterrows():
            if str(row["en"]).lower() in str(row[args.target_lang]).lower():
                drop_idx.append(i)
        if drop_idx:
            df_merged.drop(drop_idx, inplace=True)
    df_merged = df_merged.reset_index(drop=True)

    # Load tokenizer from FIRST checkpoint (assumes same vocab across steps)
    tok_helper = Olmo2Helper(
        model_id_or_path=ckpts[0],
        revision=args.revision,
        load_in_8bit=args.load_in_8bit,
        dtype=dtype,
    )
    tokenizer = tok_helper.tokenizer

    # Build dataset ONCE (fixed prompts across steps)
    dataset: List[Dict[str, Any]] = []
    for ind in tqdm(range(len(df_merged)), desc="Building dataset"):
        d = build_translation_prompt(
            df=df_merged,
            ind=ind,
            k=args.k_fewshot,
            lang_in=args.input_lang,
            lang_tgt=args.target_lang,
            seed=args.seed,
            tokenizer=tokenizer,
        )
        if d is None:
            continue
        dataset.append(d)
        if args.max_examples > 0 and len(dataset) >= args.max_examples:
            break

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty after filtering. Try different langs or relax filtering.")

    print(f"Dataset size: {len(dataset)} prompts")

    # Save dataset CSV once
    run_name = "translation_steps"
    save_dir = os.path.join(args.out_dir, run_name, f"{args.input_lang}_to_{args.target_lang}")
    os.makedirs(save_dir, exist_ok=True)
    pd.DataFrame(dataset).to_csv(os.path.join(save_dir, "dataset.csv"), index=False)

    # We will store: for each step, for each layer in layer_list:
    # mean & std over prompts, for EN and target
    results_en = {layer: [] for layer in layer_list}
    results_tgt = {layer: [] for layer in layer_list}
    results_en_std = {layer: [] for layer in layer_list}
    results_tgt_std = {layer: [] for layer in layer_list}

    # --- Loop over checkpoints (steps)
    for ckpt, step_label in zip(ckpts, steps):
        print(f"\n=== Running checkpoint: {ckpt} (step {step_label}) ===")
        olmo = Olmo2Helper(
            model_id_or_path=ckpt,
            revision=args.revision,
            load_in_8bit=args.load_in_8bit,
            dtype=dtype,
        )
        unemb = olmo.unemb

        # accumulate per-layer sums / sumsqs online (avoid storing [N,L])
        # we only need selected layers, but we still compute all layers then index.
        # For speed/memory, we keep only chosen layers per prompt.
        layer_sums_en = {layer: 0.0 for layer in layer_list}
        layer_sumsq_en = {layer: 0.0 for layer in layer_list}
        layer_sums_tgt = {layer: 0.0 for layer in layer_list}
        layer_sumsq_tgt = {layer: 0.0 for layer in layer_list}
        n_used = 0

        for d in tqdm(dataset, desc=f"Running model step={step_label}"):
            latents = olmo.latents_all_layers(d["prompt"])      # [L,B,T,H]
            logits = unemb(latents)                             # [L,B,T,V]
            probs = logits[:, 0, -1, :].float().softmax(dim=-1) # [L,V]

            latent_ids = torch.tensor(d["latent_token_id"], dtype=torch.long, device=probs.device)
            out_ids = torch.tensor(d["out_token_id"], dtype=torch.long, device=probs.device)

            p_en_all = probs[:, latent_ids].sum(dim=-1)  # [L]
            p_tgt_all = probs[:, out_ids].sum(dim=-1)    # [L]

            for layer in layer_list:
                if layer < 0 or layer >= p_en_all.shape[0]:
                    continue
                pe = float(p_en_all[layer].detach().cpu())
                pt = float(p_tgt_all[layer].detach().cpu())
                layer_sums_en[layer] += pe
                layer_sumsq_en[layer] += pe * pe
                layer_sums_tgt[layer] += pt
                layer_sumsq_tgt[layer] += pt * pt

            n_used += 1

        # finalize mean/std per layer at this step
        for layer in layer_list:
            if n_used <= 1:
                mean_en, std_en = 0.0, 0.0
                mean_tgt, std_tgt = 0.0, 0.0
            else:
                mean_en = layer_sums_en[layer] / n_used
                var_en = max(layer_sumsq_en[layer] / n_used - mean_en * mean_en, 0.0)
                std_en = float(np.sqrt(var_en))

                mean_tgt = layer_sums_tgt[layer] / n_used
                var_tgt = max(layer_sumsq_tgt[layer] / n_used - mean_tgt * mean_tgt, 0.0)
                std_tgt = float(np.sqrt(var_tgt))

            results_en[layer].append(mean_en)
            results_en_std[layer].append(std_en)
            results_tgt[layer].append(mean_tgt)
            results_tgt_std[layer].append(std_tgt)

    # --- Plot: one figure per layer, x = step
    x = np.arange(len(steps))
    x_labels = steps

    for layer in layer_list:
        en_mean = np.array(results_en[layer], dtype=np.float32)
        en_std = np.array(results_en_std[layer], dtype=np.float32)
        tg_mean = np.array(results_tgt[layer], dtype=np.float32)
        tg_std = np.array(results_tgt_std[layer], dtype=np.float32)

        en_lo, en_hi = gaussian_ci(en_mean, en_std, n=len(dataset))
        tg_lo, tg_hi = gaussian_ci(tg_mean, tg_std, n=len(dataset))

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(x, en_mean, marker="o", label="en")
        ax.fill_between(x, en_lo, en_hi, alpha=0.2)

        ax.plot(x, tg_mean, marker="o", label=args.target_lang)
        ax.fill_between(x, tg_lo, tg_hi, alpha=0.2)

        ax.set_title(f"Translation: {args.input_lang}→{args.target_lang} | layer {layer}")
        ax.set_xlabel("training step")
        ax.set_ylabel("probability")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left")
        fig.tight_layout()

        out_path = os.path.join(save_dir, f"layer_{layer}.pdf")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")

    # save raw numbers
    out_json = os.path.join(save_dir, "summary.json")
    summary = {
        "steps": steps,
        "layers": layer_list,
        "n_prompts": len(dataset),
        "en_mean": {str(k): results_en[k] for k in layer_list},
        "en_std": {str(k): results_en_std[k] for k in layer_list},
        f"{args.target_lang}_mean": {str(k): results_tgt[k] for k in layer_list},
        f"{args.target_lang}_std": {str(k): results_tgt_std[k] for k in layer_list},
    }
    import json
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary: {out_json}")


if __name__ == "__main__":
    main()
