"""
Download TinyStories, tokenize with GPT-2 byte-level BPE, and save the train /
val splits as contiguous int32 .bin files.

Run once:
    python data_prep.py --out_dir ./tiny_bin --tokenizer gpt2
"""

import argparse, os, struct, numpy as np, torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast

def encode_and_save(split, ds, tok, out_path):
    ids = []
    for txt in ds[split]["text"]:
        ids.extend(tok.encode(txt, add_special_tokens=False) + [tok.eos_token_id])
    arr = np.array(ids, dtype=np.uint32)
    arr.tofile(out_path)
    print(f"{split}: {len(arr):,} tokens → {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--tokenizer", type=str, default="gpt2")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ds  = load_dataset("roneneldan/TinyStories", split="train")
    ds  = ds.train_test_split(test_size=0.01, seed=42)
    tok = GPT2TokenizerFast.from_pretrained(args.tokenizer)

    encode_and_save("train", ds["train"], tok, os.path.join(args.out_dir, "train.bin"))
    encode_and_save("val",   ds["test"],  tok, os.path.join(args.out_dir, "val.bin"))
    # Save tokenizer config for re-use
    tok.save_pretrained(args.out_dir)

if __name__ == "__main__":
    main()
