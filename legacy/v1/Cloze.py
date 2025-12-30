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

# from original repo
from utils import plot_ci, plot_ci_plus_heatmap


# -----------------------------
# Helper: OLMo-2 wrapper
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
                    "Could not find final norm module. Tried: model.norm, final_layer_norm, final_layernorm, ln_f."
                )

        self.unemb = nn.Sequential(self.final_norm, self.lm_head)

    @torch.no_grad()
    def latents_all_layers(self, prompt: str) -> torch.Tensor:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # (emb, layer1..layerN)
        hs = hs[1:]             # drop embedding output
        return torch.stack(hs, dim=0)  # [L, B, T, H]


# -----------------------------
# Utils
# -----------------------------
lang2name = {"fr": "Français", "de": "Deutsch", "ru": "Русский", "en": "English", "zh": "中文"}


def process_tokens(token_str: str, tokenizer) -> List[int]:
    """Tokenizer-agnostic: returns unique token ids for the string."""
    ids = tokenizer.encode(str(token_str), add_special_tokens=False)
    return list(set(ids))


def compute_entropy(probas: torch.Tensor) -> torch.Tensor:
    """probas: [L, V] or [..., V]"""
    eps = 1e-12
    return (-probas * torch.log2(probas.clamp_min(eps))).sum(dim=-1)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Cloze latent-language analysis (OLMo-2 / HF)")

    # model
    ap.add_argument("--model_id_or_path", default="allenai/OLMo-2-1124-7B",
                    help="HF model id (hub) or local HF checkpoint dir")
    ap.add_argument("--revision", default=None, help="HF revision for hub models")
    ap.add_argument("--load_in_8bit", action="store_true", help="Load in 8-bit (bitsandbytes required)")
    ap.add_argument("--no_load_in_8bit", dest="load_in_8bit", action="store_false",
                    help="Disable 8-bit loading (use bf16/fp16/fp32)")
    ap.set_defaults(load_in_8bit=True)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                    help="dtype when not using 8-bit")

    # data
    ap.add_argument("--target_lang", default="fr", help="Target language (e.g., fr)")
    ap.add_argument("--prefix", default="./data/langs/", help="Prefix containing <lang>/clean.csv")
    ap.add_argument("--key", default="blank_prompt_translation_masked",
                    help="Column key in clean.csv to use as cloze prompt template")
    ap.add_argument("--n_skip", type=int, default=2, help="How many example prompts to prepend (few-shot count)")
    ap.add_argument("--max_items", type=int, default=None, help="Cap dataset size for debugging (e.g., 200)")

    # filtering
    ap.add_argument("--avoid_latent_overlap", action="store_true", default=True,
                    help="Skip examples if target token ids overlap with English token ids (default True)")
    ap.add_argument("--no_avoid_latent_overlap", dest="avoid_latent_overlap", action="store_false",
                    help="Do not filter overlapping token ids")

    # output
    ap.add_argument("--out_dir", default="./visuals", help="Output root directory")
    ap.add_argument("--tik_step", type=int, default=5, help="Tick step for plotting")

    # misc
    ap.add_argument("--seed", type=int, default=42, help="Random seed")

    return ap.parse_args()


def make_query_prefix_from_row(prompt_row: str, target_lang: str) -> str:
    """
    Original notebook used language-specific split for zh:
      if zh: split("：")[0] + ': "'
      else: split(":")[0] + ': "'
    We keep same behavior for compatibility.
    """
    if target_lang == "zh":
        return prompt_row.split("：")[0] + ': "'
    return prompt_row.split(":")[0] + ': "'


def main() -> None:
    args = parse_args()

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # dtype
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    # load model
    olmo = Olmo2Helper(
        model_id_or_path=args.model_id_or_path,
        revision=args.revision,
        load_in_8bit=args.load_in_8bit,
        dtype=dtype,
    )
    tokenizer = olmo.tokenizer
    unemb = olmo.unemb

    print(f"Loaded model: {args.model_id_or_path}")
    if args.revision:
        print(f"  revision: {args.revision}")

    # load data
    df = pd.read_csv(os.path.join(args.prefix, args.target_lang, "clean.csv"))
    if args.key not in df.columns:
        raise KeyError(
            f"Column '{args.key}' not found in CSV. Available columns: {list(df.columns)}"
        )

    # build U_normalized & avgUU for energy
    with torch.no_grad():
        U = list(unemb[1].parameters())[0].detach().cpu().float()         # [V, H]
        weights = list(unemb[0].parameters())[0].detach().cpu().float()   # [H]
        U_weighted = U * weights.unsqueeze(0)
        U_normalized = U_weighted / ((U_weighted**2).sum(dim=1, keepdim=True)).sqrt()

        v = U.shape[0]
        avgUU = (((U_normalized.T @ U_normalized) ** 2).sum() / (v**2)).sqrt()
        print(f"U {U.shape} | avgUU={avgUU.item():.6f}")

    # build cloze dataset
    dataset_gap: List[Dict[str, Any]] = []
    n = len(df)

    # precompute indices for faster sampling
    all_indices = np.arange(n)

    for idx, row in tqdm(df.iterrows(), total=n, desc="Building cloze dataset"):
        # sample n_skip example indices excluding idx
        # (deterministic via np.random seed)
        valid = np.delete(all_indices, idx)
        if len(valid) < args.n_skip:
            continue

        idx_examples = np.random.choice(valid, args.n_skip, replace=False)

        prompt_template = ""
        for ex in idx_examples:
            prompt_template += f"{df.loc[ex, args.key]}\n"

        out_token_str = row["word_translation"]   # target language word
        latent_token_str = row["word_original"]   # English word (latent)

        out_token_id = process_tokens(out_token_str, tokenizer)
        latent_token_id = process_tokens(latent_token_str, tokenizer)

        if len(out_token_id) == 0 or len(latent_token_id) == 0:
            continue

        if args.avoid_latent_overlap and args.target_lang != "en":
            if len(set(out_token_id).intersection(set(latent_token_id))) > 0:
                continue

        prompt_prefix = make_query_prefix_from_row(str(row[args.key]), args.target_lang)

        dataset_gap.append(
            {
                "prompt": prompt_template + prompt_prefix,
                "out_token_id": out_token_id,
                "out_token_str": out_token_str,
                "latent_token_id": latent_token_id,
                "latent_token_str": latent_token_str,
            }
        )

        if args.max_items is not None and len(dataset_gap) >= args.max_items:
            break

    print(f"dataset_gap size: {len(dataset_gap)}")
    if len(dataset_gap) == 0:
        raise RuntimeError("No examples left after filtering. Try --no_avoid_latent_overlap or check your CSV.")

    df_gap = pd.DataFrame(dataset_gap)
    print("Example prompt:\n", df_gap["prompt"].iloc[0])

    # output directory (safe if model path is absolute)
    run_name = os.path.basename(os.path.normpath(args.model_id_or_path))
    save_dir = os.path.join(args.out_dir, run_name, "cloze")
    os.makedirs(save_dir, exist_ok=True)

    # save dataset
    dataset_csv = os.path.join(save_dir, f"{args.target_lang}_dataset.csv")
    df_gap.to_csv(dataset_csv, index=False)
    print(f"Saved dataset CSV: {dataset_csv}")

    # run analysis
    latent_token_probs = []
    out_token_probs = []
    entropy = []
    energy = []
    latents_all = []

    for d in tqdm(dataset_gap, desc="Running model"):
        prompt = d["prompt"]

        latents = olmo.latents_all_layers(prompt)         # [L, B, T, H]
        logits = unemb(latents)                           # [L, B, T, V]
        last = logits[:, 0, -1, :].float().softmax(dim=-1).detach().cpu()  # [L, V]

        latent_ids = torch.tensor(d["latent_token_id"], dtype=torch.long)
        out_ids = torch.tensor(d["out_token_id"], dtype=torch.long)

        latent_token_probs.append(last[:, latent_ids].sum(dim=-1))  # [L]
        out_token_probs.append(last[:, out_ids].sum(dim=-1))        # [L]
        entropy.append(compute_entropy(last))                       # [L]

        # final-position latents [L, H] (batch=0, last token)
        lat_last = latents[:, 0, -1, :].float()  # [L, H]
        latents_all.append(lat_last.detach().cpu().clone())

        # energy
        lat_norm = lat_last
        lat_norm = lat_norm / (((lat_norm**2).mean(dim=-1, keepdim=True)) ** 0.5)
        lat_norm = lat_norm / (lat_norm.norm(dim=-1, keepdim=True) + 1e-12)

        proj = U_normalized.to(lat_norm.device) @ lat_norm.T  # [V, L]
        norm_val = (proj.pow(2).mean(dim=0)).sqrt()           # [L]
        energy.append((norm_val / avgUU.to(lat_norm.device)).detach().cpu())

    latent_token_probs = torch.stack(latent_token_probs)  # [N, L]
    out_token_probs = torch.stack(out_token_probs)        # [N, L]
    entropy = torch.stack(entropy)                        # [N, L]
    energy = torch.stack(energy)                          # [N, L]
    latents = torch.stack(latents_all)                    # [N, L, H]

    print("latents:", latents.shape)
    print("latent_token_probs:", latent_token_probs.shape)
    print("out_token_probs:", out_token_probs.shape)

    # plots
    fig, ax, ax2 = plot_ci_plus_heatmap(
        latent_token_probs,
        entropy,
        "en",
        color="tab:blue",
        tik_step=args.tik_step,
        do_colorbar=True,
        nums=[0.99, 0.18, 0.025, 0.6],
    )

    if args.target_lang != "en":
        plot_ci(ax2, out_token_probs, args.target_lang, color="tab:orange", do_lines=False)

    ax2.set_xlabel("layer")
    ax2.set_ylabel("probability")
    ax2.set_xlim(0, out_token_probs.shape[1] + 1)
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left")

    plot1_path = os.path.join(save_dir, f"olmo2_{args.target_lang}_probas_ent.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {plot1_path}")

    fig2, axE = plt.subplots(figsize=(5, 3))
    plot_ci(axE, energy, "energy", color="tab:green", do_lines=True, tik_step=args.tik_step)
    axE.set_xlabel("layer")
    axE.set_ylabel("energy")
    axE.set_xlim(0, out_token_probs.shape[1] + 1)

    plot2_path = os.path.join(save_dir, f"olmo2_{args.target_lang}_energy.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {plot2_path}")

    # save latents
    latents_path = os.path.join(save_dir, f"olmo2_{args.target_lang}_latents.pt")
    torch.save(latents, latents_path)
    print(f"Saved latents: {latents_path}")


if __name__ == "__main__":
    main()
