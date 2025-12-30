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

from utils import plot_ci


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Olmo2Helper:
    def __init__(
        self,
        model_id_or_path: str,
        revision: Optional[str] = None,
        load_in_8bit: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, revision=revision, use_fast=True)
        if load_in_8bit:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id_or_path, revision=revision, device_map="auto", load_in_8bit=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id_or_path, revision=revision, device_map="auto", torch_dtype=dtype
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
                raise AttributeError("Could not find final norm module (model.norm / ln_f / etc.)")

        self.unemb = nn.Sequential(self.final_norm, self.lm_head)

    @torch.no_grad()
    def latents_all_layers(self, prompt: str) -> torch.Tensor:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[1:]  # drop embedding output
        return torch.stack(hs, dim=0)  # [L,B,T,H]


def process_tokens(token_str: str, tokenizer) -> List[int]:
    ids = tokenizer.encode(str(token_str), add_special_tokens=False)
    return list(set(ids))


def gaussian_ci(mean: np.ndarray, std: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n <= 1:
        return mean, mean
    half = 1.96 * (std / np.sqrt(n))
    return mean - half, mean + half


def parse_int_list(s: str) -> List[int]:
    if s.strip() == "":
        return []
    return [int(x.strip()) for x in s.split(",")]


def build_cloze_dataset(
    df: pd.DataFrame,
    target_lang: str,
    tokenizer,
    key: str,
    n_examples: int,
    seed: int,
    max_examples: int = 0,
) -> List[Dict[str, Any]]:
    """
    Reproduces notebook logic:
      - sample n_examples other masked prompts as demonstrations
      - query prompt is '<Lang>: "' (open quote) after the colon
      - out token = word_translation
      - latent token = word_original (English)
      - skip overlap if target != en
    """
    dataset = []
    df = df.reset_index(drop=True)

    for idx in tqdm(range(len(df)), desc="Building cloze dataset"):
        rng = np.random.RandomState(seed + idx)

        # pick demonstrations
        indices = np.arange(len(df))
        indices = indices[indices != idx]
        if len(indices) < n_examples:
            continue
        ex_ids = rng.choice(indices, size=n_examples, replace=False)

        prompt = ""
        for ex_i in ex_ids:
            prompt += str(df.loc[ex_i, key]) + "\n"

        row = df.loc[idx]
        out_token_str = row["word_translation"]   # target word
        latent_token_str = row["word_original"]   # english word

        out_token_id = process_tokens(out_token_str, tokenizer)
        latent_token_id = process_tokens(latent_token_str, tokenizer)

        if len(out_token_id) == 0 or len(latent_token_id) == 0:
            continue
        if target_lang != "en" and len(set(out_token_id).intersection(set(latent_token_id))) > 0:
            continue

        # build the query prefix like notebook:
        # if zh uses Chinese colon "：", else ":"
        masked = str(row[key])
        if target_lang == "zh":
            query_prefix = masked.split("：")[0] + ': "'
        else:
            query_prefix = masked.split(":")[0] + ': "'

        dataset.append(
            {
                "prompt": prompt + query_prefix,
                "out_token_id": out_token_id,
                "latent_token_id": latent_token_id,
                "out_token_str": out_token_str,
                "latent_token_str": latent_token_str,
            }
        )

        if max_examples > 0 and len(dataset) >= max_examples:
            break

    return dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_lang", type=str, default="fr")
    ap.add_argument("--data_prefix", type=str, required=True, help="Path containing <lang>/clean.csv")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--checkpoints", type=str, required=True,
                    help="Comma-separated list of checkpoint paths (HF format) OR a single HF hub id.")
    ap.add_argument("--steps", type=str, required=True,
                    help="Comma-separated list of step labels (same length as checkpoints), e.g. 477,4770,8290")

    ap.add_argument("--revision", type=str, default=None)
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--no_load_in_8bit", dest="load_in_8bit", action="store_false")
    ap.set_defaults(load_in_8bit=True)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    ap.add_argument("--layers", type=str, default="10,20,30")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_examples", type=int, default=2, help="Number of demonstration cloze lines (not counting the query).")
    ap.add_argument("--key", type=str, default="blank_prompt_translation_masked")
    ap.add_argument("--max_examples", type=int, default=0)

    args = ap.parse_args()
    set_seed(args.seed)

    ckpts = [x.strip() for x in args.checkpoints.split(",") if x.strip()]
    steps = [x.strip() for x in args.steps.split(",") if x.strip()]
    if len(ckpts) != len(steps):
        raise ValueError(f"--checkpoints and --steps must have same length. Got {len(ckpts)} vs {len(steps)}")

    layer_list = parse_int_list(args.layers)
    if len(layer_list) == 0:
        raise ValueError("--layers must be non-empty, e.g. 10,20,30")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Load data
    df = pd.read_csv(os.path.join(args.data_prefix, args.target_lang, "clean.csv")).reset_index(drop=True)

    # Load tokenizer from first checkpoint (assume same vocab across steps)
    tok_helper = Olmo2Helper(
        model_id_or_path=ckpts[0],
        revision=args.revision,
        load_in_8bit=args.load_in_8bit,
        dtype=dtype,
    )
    tokenizer = tok_helper.tokenizer

    # Build dataset once
    dataset = build_cloze_dataset(
        df=df,
        target_lang=args.target_lang,
        tokenizer=tokenizer,
        key=args.key,
        n_examples=args.n_examples,
        seed=args.seed,
        max_examples=args.max_examples,
    )
    if len(dataset) == 0:
        raise RuntimeError("Cloze dataset is empty after filtering. Try different target_lang or adjust filtering.")

    print(f"Dataset size: {len(dataset)} prompts")

    # Save dataset CSV once
    run_name = "cloze_steps"
    save_dir = os.path.join(args.out_dir, run_name, args.target_lang)
    os.makedirs(save_dir, exist_ok=True)
    pd.DataFrame(dataset).to_csv(os.path.join(save_dir, "dataset.csv"), index=False)

    results_en = {layer: [] for layer in layer_list}
    results_tgt = {layer: [] for layer in layer_list}
    results_en_std = {layer: [] for layer in layer_list}
    results_tgt_std = {layer: [] for layer in layer_list}

    # Loop checkpoints
    for ckpt, step_label in zip(ckpts, steps):
        print(f"\n=== Running checkpoint: {ckpt} (step {step_label}) ===")
        olmo = Olmo2Helper(
            model_id_or_path=ckpt,
            revision=args.revision,
            load_in_8bit=args.load_in_8bit,
            dtype=dtype,
        )
        unemb = olmo.unemb

        layer_sums_en = {layer: 0.0 for layer in layer_list}
        layer_sumsq_en = {layer: 0.0 for layer in layer_list}
        layer_sums_tgt = {layer: 0.0 for layer in layer_list}
        layer_sumsq_tgt = {layer: 0.0 for layer in layer_list}
        n_used = 0

        for d in tqdm(dataset, desc=f"Running model step={step_label}"):
            latents = olmo.latents_all_layers(d["prompt"])          # [L,B,T,H]
            logits = unemb(latents)                                 # [L,B,T,V]
            probs = logits[:, 0, -1, :].float().softmax(dim=-1)     # [L,V]

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

    # Plot per layer (x = step)
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

        # ax.set_title(f"Cloze: target={args.target_lang} | layer {layer}")
        ax.set_xlabel("training step")
        ax.set_ylabel("probability")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left")
        fig.tight_layout()

        out_path = os.path.join(save_dir, f"layer_{layer}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")

    # Save raw numbers
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
