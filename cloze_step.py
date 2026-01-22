import argparse
import os
import json
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import plot_ci

from transformers import AutoTokenizer, AutoModelForCausalLM

from collections import Counter


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# OLMo-2 Helper (drop-in replacement for LlamaHelper)
# -----------------------------
class Olmo2Helper:
    def __init__(
        self,
        dir: str,
        revision: Optional[str] = None,
        load_in_8bit: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ):
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

        self.lm_head = self.model.get_output_embeddings()
        self.unemb = nn.Sequential(self.final_norm, self.lm_head)

    @torch.no_grad()
    def latents_all_layers(self, prompt: str) -> torch.Tensor:
        """
        Return hidden states per layer: [L, T, H] (batch squeezed)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[1:]                # drop embedding output
        return torch.stack(hs, dim=0)[:, 0, :, :] # [L,T,H]


# -----------------------------
# Notebook token logic (unchanged)
# -----------------------------
def token_prefixes(token_str: str):
    n = len(token_str)
    return [token_str[:i] for i in range(1, n + 1)]

def add_spaces(tokens):
    # return ["▁" + t for t in tokens] + tokens
    return ['Ġ' + t for t in tokens] + ["▁" + t for t in tokens] + tokens

def capitalizations(tokens):
    return list(set(tokens))

def _gpt2_bytes_to_unicode():
    """
    GPT-2/OLMo-2 byte->unicode mapping used by byte-level BPE.
    Returns dict: int byte (0..255) -> str (single unicode char).
    """
    bs = list(range(ord("!"), ord("~") + 1)) \
       + list(range(ord("¡"), ord("¬") + 1)) \
       + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

_BYTE2UNI = _gpt2_bytes_to_unicode()

def olmo_unicode_prefix_tokid(zh_char="云", tokenizer=None, verbose=True):
    """
    OLMo-2 version of "first byte token id":
    - Take UTF-8 first byte of the character
    - Map that byte to GPT-2/OLMo-2 byte-unicode symbol
    - Look up that symbol in the tokenizer vocab
    """
    if tokenizer is None:
        return None
    try:
        b = zh_char.encode("utf-8")
        first = b[0]  # int 0..255
        key = _BYTE2UNI[first]  # the vocab string for that byte
        if verbose:
            print("char:", zh_char)
            print("utf8 bytes:", [f"0x{x:02X}" for x in b])
            print("first byte:", f"0x{first:02X}")
            print("vocab key repr:", repr(key))
        vocab = tokenizer.get_vocab()
        tid = vocab.get(key, None)
        if verbose:
            print("token_id:", tid)
        return tid
    except Exception as e:
        if verbose:
            print("error:", repr(e))
        return None

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

def process_tokens(token_str: str, tokenizer, lang: str):
    token_str = str(token_str)
    with_prefixes = token_prefixes(token_str)
    with_spaces = add_spaces(with_prefixes)
    with_caps = capitalizations(with_spaces)

    vocab = tokenizer.get_vocab()
    final_tokens = [vocab[tok] for tok in with_caps if tok in vocab]

    if lang in ["zh", "ru"]:
        tokid = olmo_unicode_prefix_tokid(token_str, tokenizer)
        if tokid is not None:
            final_tokens.append(tokid)

    ids = tokenizer.encode(token_str, add_special_tokens=False)
    if ids:
        final_tokens.extend(ids)

    return list(set(final_tokens))


# -----------------------------
# Dataset construction (same as notebook)
# -----------------------------
def build_dataset_gap(
    df: pd.DataFrame,
    target_lang: str,
    tokenizer,
    key: str,
    n_skip: int,
    seed: int,
    max_examples: int = 0,
) -> List[Dict[str, Any]]:
    dataset_gap = []
    df = df.reset_index(drop=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building dataset_gap"):
        rng = np.random.RandomState(seed + idx)

        indices = np.arange(len(df))
        indices = indices[indices != idx]
        if len(indices) < n_skip:
            continue
        idx_examples = rng.choice(indices, n_skip, replace=False)

        prompt_template = ""
        for ex in idx_examples:
            prompt_template += f"{df.loc[ex, key]}\n"

        out_token_str = row["word_translation"]
        latent_token_str = row["word_original"]

        out_token_id = process_tokens(out_token_str, tokenizer, target_lang)
        latent_token_id = process_tokens(latent_token_str, tokenizer, "en")

        if len(out_token_id) == 0 or len(latent_token_id) == 0:
            continue
        if target_lang != "en" and len(set(out_token_id).intersection(set(latent_token_id))) > 0:
            continue

        masked = str(row[key])
        if target_lang == "zh":
            prompt = masked.split("：")[0] + '：\"'
        else:
            prompt = masked.split(":")[0] + ': \"'

        actual_out_token_str_id = tokenizer.encode(out_token_str, add_special_tokens=False)
        # list of readable tokens from out_token_id
        list_out_token_str = [tokenizer.convert_ids_to_tokens(tid) for tid in actual_out_token_str_id]

        # list of readable tokens from latent_token_id
        list_latent_token_str = [tokenizer.convert_ids_to_tokens(tid) for tid in latent_token_id]

        dataset_gap.append(
            {
                "prompt": prompt_template + prompt,
                "out_token_id": out_token_id,
                "actual_out_token_str_id": actual_out_token_str_id,
                "list_actual_out_token_str": list_out_token_str,
                "out_token_str": out_token_str,
                "latent_token_id": latent_token_id,
                "list_latent_token_str": list_latent_token_str,
                "latent_token_str": latent_token_str,
            }
        )

        if max_examples > 0 and len(dataset_gap) >= max_examples:
            break

    return dataset_gap


def decode_token_ids(tokenizer, ids):
    return [(int(i), tokenizer.convert_ids_to_tokens(int(i))) for i in ids]


def top_argmax_summary(argmax_ids: list[int], tokenizer, topk: int = 10):
    c = Counter(argmax_ids)
    top = c.most_common(topk)
    return [
        {"id": tid, "tok": tokenizer.convert_ids_to_tokens(tid), "count": cnt}
        for tid, cnt in top
    ]

# -----------------------------
# Main
# -----------------------------
def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def step_to_B(s):
    s = s.strip()
    if s.endswith("B"):
        return float(s[:-1])
    if s.endswith("M"):
        return float(s[:-1]) / 1000.0
    return float(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_lang", type=str, required=True)
    ap.add_argument("--data_prefix", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--checkpoints", type=str, required=True,
                    help="Comma-separated list of HF checkpoint paths or hub ids.")
    ap.add_argument("--steps", type=str, required=True,
                    help="Comma-separated list of step labels, same length as checkpoints.")

    ap.add_argument("--layers", type=str, default="10,20,30")
    ap.add_argument("--key", type=str, default="blank_prompt_translation_masked")
    ap.add_argument("--n_skip", type=int, default=2)
    ap.add_argument("--max_examples", type=int, default=0)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--revision", type=str, default=None)
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--no_load_in_8bit", dest="load_in_8bit", action="store_false")
    ap.set_defaults(load_in_8bit=True)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    ap.add_argument("--img_ext", type=str, default="png", choices=["png", "jpg", "jpeg", "svg"])
    ap.add_argument("--dpi", type=int, default=300)

    args = ap.parse_args()
    set_seed(args.seed)

    ckpts = [x.strip() for x in args.checkpoints.split(",") if x.strip()]
    steps = [x.strip() for x in args.steps.split(",") if x.strip()]
    if len(ckpts) != len(steps):
        raise ValueError("--checkpoints and --steps must have the same length")

    layer_list = parse_int_list(args.layers)
    if not layer_list:
        raise ValueError("--layers must be non-empty")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Load data
    df = pd.read_csv(os.path.join(args.data_prefix, args.target_lang, "clean.csv")).reset_index(drop=True)

    # Build tokenizer from the first checkpoint, then dataset once (close to notebook)
    helper0 = Olmo2Helper(ckpts[0], revision=args.revision, load_in_8bit=args.load_in_8bit, dtype=dtype)
    tokenizer = helper0.tokenizer

    dataset_gap = build_dataset_gap(
        df=df,
        target_lang=args.target_lang,
        tokenizer=tokenizer,
        key=args.key,
        n_skip=args.n_skip,
        seed=args.seed,
        max_examples=args.max_examples,
    )
    if len(dataset_gap) == 0:
        raise RuntimeError("dataset_gap is empty after filtering.")

    # Output dir
    save_dir = os.path.join(args.out_dir, "cloze_steps", args.target_lang)
    os.makedirs(save_dir, exist_ok=True)
    pd.DataFrame(dataset_gap).to_csv(os.path.join(save_dir, "dataset.csv"), index=False)

    per_layer_en_by_step = {layer: [] for layer in layer_list}
    per_layer_tgt_by_step = {layer: [] for layer in layer_list}
    per_layer_argmax_by_step = {layer: [] for layer in layer_list}  # per step -> list[prompt argmax ids]

    # Loop checkpoints
    for ckpt, step in zip(ckpts, steps):
        print(f"\n=== Step {step}: {ckpt} ===")
        olmo = Olmo2Helper(ckpt, revision=args.revision, load_in_8bit=args.load_in_8bit, dtype=dtype)
        unemb = olmo.unemb

        # collect per-prompt values for selected layers (stable + simple)
        step_vals_en = {layer: [] for layer in layer_list}
        step_vals_tgt = {layer: [] for layer in layer_list}
        step_argmax_ids = {layer: [] for layer in layer_list}

        for d in tqdm(dataset_gap, desc=f"Running step={step}"):
            latents = olmo.latents_all_layers(d["prompt"])   # [L,T,H]
            logits = unemb(latents)                          # [L,T,V]

            last = logits[:, -1, :].float().softmax(dim=-1)  # [L,V]
            argmax_all = last.argmax(dim=-1)

            latent_ids = torch.tensor(d["latent_token_id"], dtype=torch.long, device=last.device)
            out_ids = torch.tensor(d["out_token_id"], dtype=torch.long, device=last.device)

            p_en_all = last[:, latent_ids].sum(dim=-1)   # [L]
            p_tg_all = last[:, out_ids].sum(dim=-1)      # [L]

            for layer in layer_list:
                if 0 <= layer < p_en_all.shape[0]:
                    step_vals_en[layer].append(float(p_en_all[layer].detach().cpu()))
                    step_vals_tgt[layer].append(float(p_tg_all[layer].detach().cpu()))
                    step_argmax_ids[layer].append(int(argmax_all[layer].detach().cpu()))

        # aggregate per step
        for layer in layer_list:
            per_layer_en_by_step[layer].append(step_vals_en[layer])     # list length N
            per_layer_tgt_by_step[layer].append(step_vals_tgt[layer])
            per_layer_argmax_by_step[layer].append(step_argmax_ids[layer])  # [N]

        print(f"\n[Argmax summary] step={step}")
        argmax_token_data = []
        for layer in layer_list:
            ids = step_argmax_ids[layer]
            if len(ids) == 0:
                print(f"  layer {layer}: (no data)")
                continue
            top5 = top_argmax_summary(ids, tokenizer, topk=5)
            msg = ", ".join([f"{d['tok']} - (tok_id: {d['id']}) - (count: {d['count']})" for d in top5])
            print(f"  layer {layer}: {msg}")

            argmax_token_data.append({
                "layer": layer,
                "top_tokens": top5,
            })
        # Save per-step argmax summary
        with open(os.path.join(save_dir, f"argmax_summary_step_{step}.json"), "w", encoding="utf-8") as f:
            json.dump(argmax_token_data, f, indent=2, ensure_ascii=False)
        print("Saved:", os.path.join(save_dir, f"argmax_summary_step_{step}.json"))

    # Plot (one chart per layer): x = step
    # x = np.arange(len(steps))

    dump = {
        "steps": steps,
        "layers": layer_list,
        "n_prompts": len(dataset_gap),
        "checkpoints": ckpts,
        "target_lang": args.target_lang,
        "per_layer_en_by_step": per_layer_en_by_step,
        "per_layer_tgt_by_step": per_layer_tgt_by_step,
        "per_layer_argmax_by_step": per_layer_argmax_by_step,
    }

    dump_path = os.path.join(save_dir, "raw_dump.json")
    with open(dump_path, "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2, ensure_ascii=False)

    print("Saved:", dump_path)

    S = len(steps)
    x_vals = [step_to_B(s) for s in steps]

    for layer in layer_list:
        en_mat_np = np.array(per_layer_en_by_step[layer], dtype=np.float32).T
        tg_mat_np = np.array(per_layer_tgt_by_step[layer], dtype=np.float32).T

        en_mat = torch.tensor(en_mat_np, dtype=torch.float32)
        tg_mat = torch.tensor(tg_mat_np, dtype=torch.float32)

        fig, ax = plt.subplots(figsize=(6.2, 3.6))

        plot_ci(ax, en_mat, "en", tik_step=1, do_lines=False, x=x_vals, color="orange")
        plot_ci(ax, tg_mat, args.target_lang, tik_step=1, do_lines=False, x=x_vals, color="blue")

        # ax.set_title(f"Cloze | target={args.target_lang} | layer {layer + 1}")
        # ax.set_title(f"Cloze | target={args.target_lang}")
        
        ax.set_xlabel("# training tokens", fontsize="large")
        ax.set_ylabel("probability", fontsize="large")
        ax.set_ylim(0, 0.5)
        # ax.set_xticks(np.arange(1, S + 1))
        ax.set_xticks(x_vals)
        ax.set_xticklabels(steps, rotation=30, ha="right")
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
        "n_prompts": len(dataset_gap),
    }
    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Saved:", os.path.join(save_dir, "summary.json"))


if __name__ == "__main__":
    main()
