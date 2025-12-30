import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from matplotlib import pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM


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
                dtype=dtype,
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


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def token_prefixes(token_str: str):
    n = len(token_str)
    tokens = [token_str[:i] for i in range(1, n+1)]
    return tokens 

def add_spaces(tokens):
    return ['▁' + t for t in tokens] + tokens

def capitalizations(tokens):
    return list(set(tokens))

def unicode_prefix_tokid(zh_char = "云", tokenizer=None):
    if tokenizer is None:
        return None

    start = zh_char.encode().__str__()[2:-1].split('\\x')[1]
    unicode_format = '<0x%s>'
    start_key = unicode_format%start.upper()
    if start_key in tokenizer.get_vocab():
        return tokenizer.get_vocab()[start_key]
    return None

def compute_entropy(probas):
        return (-probas*torch.log2(probas)).sum(dim=-1)

def process_tokens(token_str: str, tokenizer, lang):
    with_prefixes = token_prefixes(token_str)
    with_spaces = add_spaces(with_prefixes)
    with_capitalizations = capitalizations(with_spaces)
    final_tokens = []
    for tok in with_capitalizations:
        if tok in tokenizer.get_vocab():
            final_tokens.append(tokenizer.get_vocab()[tok])
    if lang in ['zh', 'ru']:
        tokid = unicode_prefix_tokid(token_str, tokenizer)
        if tokid is not None:
            final_tokens.append(tokid)
    return final_tokens

def get_tokens(token_ids, olmo2_helper):
    id2voc = {id:voc for voc, id in olmo2_helper.tokenizer.get_vocab().items()}
    return [id2voc[tokid] for tokid in token_ids]

def unicode_prefix_tokid(zh_char = "云", tokenizer=None):
    start = zh_char.encode().__str__()[2:-1].split('\\x')[1]
    unicode_format = '<0x%s>'
    start_key = unicode_format%start.upper()
    if start_key in tokenizer.get_vocab():
        return tokenizer.get_vocab()[start_key]
    return None


def main():
    parser = argparse.ArgumentParser(description="Translation OLMO2 Script")
    
    parser.add_argument('--source_lang', type=str, required=True, help='Source language code')
    parser.add_argument('--target_lang', type=str, required=True, help='Target language code')
    parser.add_argument('--model', type=str, required=True, help='Path to the model')
    parser.add_argument("--revision", type=str, default=None)

    parser.add_argument('--data_prefix', type=str, required=True, help='Prefix path for data files')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--no_load_in_8bit", dest="load_in_8bit", action="store_false")
    parser.set_defaults(load_in_8bit=True)

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument('--img_ext', type=str, default='png', help='Image file extension')
    parser.add_argument("--dpi", type=int, default=300)

    parser.add_argument("--single_token_only", action="store_true", help="Keep only target strings that are 1 token")
    parser.add_argument("--multi_token_only", action="store_true", help="Keep only target strings with >1 token")

    args = parser.parse_args()
    set_seed(args.seed)

    single_token_only = args.single_token_only
    multi_token_only = args.multi_token_only

    input_lang = args.source_lang
    target_lang = args.target_lang

    out_dir = args.out_dir
    img_ext = args.img_ext
    dpi = args.dpi

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    df_src = pd.read_csv(f'{args.data_prefix}{args.source_lang}/clean.csv').reindex()
    df_tgt = pd.read_csv(f'{args.data_prefix}{args.target_lang}/clean.csv').reindex()

    olmo2_helper = Olmo2Helper(dir=args.model, load_in_8bit=args.load_in_8bit, revision=args.revision, dtype=dtype)

    tokenizer = olmo2_helper.tokenizer
    model = olmo2_helper.model

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

    U = list(unemb[1].parameters())[0].detach().cpu().float()
    weights = list(unemb[0].parameters())[0].detach().cpu().float()
    print(f'U {U.shape} weights {weights.unsqueeze(0).shape}')
    U_weighted = U.clone() 
    #U_weighted = U_weighted / ((U_weighted**2).mean(dim=1, keepdim=True))**0.5
    U_weighted *= weights.unsqueeze(0)
    U_normalized = U_weighted / ((U_weighted**2).sum(dim=1, keepdim=True))**0.5
    v = U.shape[0]
    TT = U_normalized.T @ U_normalized
    avgUU = (((U_normalized.T @ U_normalized)**2).sum() / v**2)**0.5
    print(avgUU.item())

    count = 0
    for idx, word in enumerate(df_tgt['word_translation']):
        if word in tokenizer.get_vocab() or '▁'+word in tokenizer.get_vocab():
            count += 1
            if multi_token_only:
                df_tgt.drop(idx, inplace=True)
        elif single_token_only:
            df_tgt.drop(idx, inplace=True)

    print(f'for {target_lang} {count} of {len(df_tgt)} are single tokens')

    if input_lang == target_lang:
        df_tgt_copy = df_tgt.copy()
        df_tgt_copy.rename(columns={'word_original': 'en', 
                                    f'word_translation': target_lang if target_lang != 'en' else 'en_tgt'}, 
                                    inplace=True)
    else:
        df_tgt_copy = df_tgt.merge(df_src, on=['word_original'], suffixes=(f'_{target_lang}', f'_{input_lang}'))
        df_tgt_copy.rename(columns={'word_original': 'en', 
                                    f'word_translation_{target_lang}': target_lang if target_lang != 'en' else 'en_tgt', 
                                    f'word_translation_{input_lang}': input_lang if input_lang != 'en' else 'en_in'}, 
                                    inplace=True)
    # delete all rows where en is contained in de or fr
    if target_lang != 'en':
        for i, row in df_tgt_copy.iterrows():
            if row['en'].lower() in row[target_lang].lower():
                df_tgt_copy.drop(i, inplace=True)

    print(f'final length of df_tgt_copy: {len(df_tgt_copy)}')

    def sample(df, ind, k=5, tokenizer=None, lang1='fr', lang2='de', lang_latent='en'):
        lang2name = {'fr': 'Français', 'de': 'Deutsch', 'ru': 'Русский', 'en': 'English', 'zh': '中文'}
        df = df.reset_index(drop=True)
        temp = df[df.index!=ind]
        sample = pd.concat([temp.sample(k-1), df[df.index==ind]], axis=0)
        prompt = ""
        for idx, (df_idx, row) in enumerate(sample.iterrows()):
            if idx < k-1:
                prompt += f'{lang2name[lang1]}: "{row[lang1]}" - {lang2name[lang2]}: "{row[lang2]}"\n'
            else:
                prompt += f'{lang2name[lang1]}: "{row[lang1]}" - {lang2name[lang2]}: "'
                in_token_str = row[lang1]
                out_token_str = row[lang2]
                out_token_id = process_tokens(out_token_str, tokenizer, lang2)
                latent_token_str = row[lang_latent]
                latent_token_id = process_tokens(latent_token_str, tokenizer, 'en')
                intersection = set(out_token_id).intersection(set(latent_token_id))
                if len(out_token_id) == 0 or len(latent_token_id) == 0:
                    yield None
                if lang2 != 'en' and len(intersection) > 0:
                    yield None
                yield {'prompt': prompt, 
                    'out_token_id': out_token_id, 
                    'out_token_str': out_token_str,
                    'latent_token_id': latent_token_id, 
                    'latent_token_str': latent_token_str, 
                    'in_token_str': in_token_str}

    dataset = []
    for ind in tqdm(range(len(df_tgt_copy))):
        d = next(sample(df=df_tgt_copy, ind=ind, tokenizer=tokenizer, lang1=input_lang, lang2=target_lang))
        if d is None:
            continue
        dataset.append(d)

    print(f'final dataset length: {len(dataset)}')

    df = pd.DataFrame(dataset)
    custom_model = args.model.split('/')[-1]
    os.makedirs(f'{os.path.join(out_dir, custom_model)}/translation', exist_ok=True)
    if single_token_only:
        df.to_csv(f'{os.path.join(out_dir, custom_model)}/translation/{input_lang}_{target_lang}_dataset_single_token.csv', index=False)
    elif multi_token_only:
        df.to_csv(f'{os.path.join(out_dir, custom_model)}/translation/{input_lang}_{target_lang}_dataset_multi_token.csv', index=False)
    else:
        df.to_csv(f'{os.path.join(out_dir, custom_model)}/translation/{input_lang}_{target_lang}_dataset.csv', index=False)

    in_token_probs = []
    latent_token_probs = []
    out_token_probs = []
    entropy = []
    energy = []
    latents_all = []

    for idx, d in tqdm(enumerate(dataset)):
        latents = llama.latents_all_layers(d['prompt'])
        logits = unemb(latents)
        last = logits[:, -1, :].float().softmax(dim=-1).detach().cpu()
        latent_token_probs += [last[:, torch.tensor(d['latent_token_id'])].sum(dim=-1)]
        out_token_probs += [last[:, torch.tensor(d['out_token_id'])].sum(dim=-1)]
        entropy += [compute_entropy(last)]
        latents_all += [latents[:, -1, :].float().detach().cpu().clone()]
        latents_normalized = latents[:, -1, :].float()
        latents_normalized = latents_normalized / (((latents_normalized**2).mean(dim=-1, keepdim=True))**0.5)
        latents_normalized /= (latents_normalized.norm(dim=-1, keepdim=True))
        norm = ((U_normalized @ latents_normalized.T)**2).mean(dim=0)**0.5
        energy += [norm/avgUU]

    latent_token_probs = torch.stack(latent_token_probs)
    out_token_probs = torch.stack(out_token_probs)
    entropy = torch.stack(entropy)
    energy = torch.stack(energy)
    latents = torch.stack(latents_all)

    fig, ax, ax2 = plot_ci_plus_heatmap(latent_token_probs, entropy, 'en', color='tab:orange', tik_step=5, do_colorbar=True, #, do_colorbar=(model_size=='70b'),
    nums=[.99, 0.18, 0.025, 0.6])
    if target_lang != 'en':
        plot_ci(ax2, out_token_probs, target_lang, color='tab:blue', do_lines=False)
    ax2.set_xlabel('layer')
    ax2.set_ylabel('probability')
    if model_size == '7b':
        ax2.set_xlim(0, out_token_probs.shape[1]+1)
    else:
        ax2.set_xlim(0, round(out_token_probs.shape[1]/10)*10+1)
    ax2.set_ylim(0, 1)
    # make xticks start from 1
    # put legend on the top left
    ax2.legend(loc='upper left')
    os.makedirs(f'{os.path.join(out_dir, custom_model)}/translation', exist_ok=True)
    if single_token_only:
        plt.savefig(f'{os.path.join(out_dir, custom_model)}/translation/{input_lang}_{target_lang}_probas_ent_single_token.{img_ext}', dpi=dpi, bbox_inches='tight')
    elif multi_token_only:
        plt.savefig(f'{os.path.join(out_dir, custom_model)}/translation/{input_lang}_{target_lang}_probas_ent_multi_token.{img_ext}', dpi=dpi, bbox_inches='tight')
    else:
        plt.savefig(f'{os.path.join(out_dir, custom_model)}/translation/{input_lang}_{target_lang}_probas_ent.{img_ext}', dpi=dpi, bbox_inches='tight')

    
    fig, ax2 = plt.subplots(figsize=(5,3))
    plot_ci(ax2, energy, 'energy', color='tab:green', do_lines=True, tik_step=5)
    ax2.set_xlabel('layer')
    ax2.set_ylabel('energy')
    if model_size == '7b':
        ax2.set_xlim(0, out_token_probs.shape[1]+1)
    else:
        ax2.set_xlim(0, round(out_token_probs.shape[1]/10)*10+1)
    os.makedirs(f'{os.path.join(out_dir, custom_model)}/translation', exist_ok=True)
    if single_token_only:
        plt.savefig(f'{os.path.join(out_dir, custom_model)}/translation/{input_lang}_{target_lang}_probas_ent_single_token.{img_ext}', dpi=dpi, bbox_inches='tight')
    elif multi_token_only:
        plt.savefig(f'{os.path.join(out_dir, custom_model)}/translation/{input_lang}_{target_lang}_probas_ent_multi_token.{img_ext}', dpi=dpi, bbox_inches='tight')
    else:
        plt.savefig(f'{os.path.join(out_dir, custom_model)}/translation/{input_lang}_{target_lang}_energy.{img_ext}', dpi=dpi, bbox_inches='tight')

    
    if single_token_only:
        torch.save(latents, f'{os.path.join(out_dir, custom_model)}/translation/{input_lang}_{target_lang}_latents_single_token.pt')
    elif multi_token_only:
        torch.save(latents, f'{os.path.join(out_dir, custom_model)}/translation/{input_lang}_{target_lang}_latents_multi_token.pt')
    else:
        torch.save(latents, f'{os.path.join(out_dir, custom_model)}/translation/{input_lang}_{target_lang}_latents.pt')


if __name__ == "__main__":
    main()