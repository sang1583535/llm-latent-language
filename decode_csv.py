import argparse
import pandas as pd
from transformers import AutoTokenizer

def decode_token_ids(tokenizer, ids):
    # show ids + decoded pieces (safe even if some ids map to special bytes)
    return [(int(i), tokenizer.convert_ids_to_tokens(int(i))) for i in ids]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--model_name", type=str, default="", help="Model name for tokenizer")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    df = pd.read_csv(args.input_csv)

    # read field out_token_id (e.g. [91386]) as list of ints
    df['out_token_id'] = df['out_token_id'].apply(lambda x: eval(x) if pd.notna(x) else [])
    # decode token ids
    df['decoded_tokens'] = df['out_token_id'].apply(lambda ids: decode_token_ids(tokenizer, ids))

    df.to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    main()