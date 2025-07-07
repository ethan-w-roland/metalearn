"""
@author: ethan-w-roland
@date: 2025-07-06
@title: Shared Backbone Meta-Learned Loss Function
"""

import argparse, itertools, json, os, torch
import torch.nn as nn
import torch.optim as optim
from model import Transformer, Config
from dataloader import create_loader

def load_vocab(token_dir: str) -> int:
    with open(os.path.join(token_dir, "tokenizer_config.json")) as f:
        return json.load(f)["vocab_size"]

def run(
        data_dir: str,
        block_size: int,
        batch_size: int,
        device: str,
        lr_inner: float,
        lr_outer: float,
) -> None:
    
    # --- Model & Loaders ---

    vocab_size  = load_vocab(data_dir)
    config = Config(vocab_size=vocab_size, block_size=block_size)
    model = Transformer(config).to(device)

    train_dl = create_loader(
        f"{data_dir}/train.bin",
        block_size,
        batch_size,
        device)
    
    val_dl = create_loader(
        f"{data_dir}/val.bin",
        block_size,
        batch_size,
        device)

    # --- Optimizers ---

    opt_inner = optim.AdamW(model.base_parameters(), lr=lr_inner)
    opt_outer = optim.AdamW(model.meta_parameters(), lr=lr_outer)

    ce_loss = nn.CrossEntropyLoss()
    loss_clip = 1.0

    for step in itertools.count(start=0):

        # --- Inner Optimizer ---

        opt_inner.zero_grad()
        data = next(train_dl)

        token_embed = model.token_embed(data)
        logits = model(token_embed, meta=False) # [B,T,V]

        logit_embed = model.logit_embed(logits)
        meta_inp = torch.cat([token_embed, logit_embed], dim=1)
        inner_loss = model(meta_inp, meta=True)
        inner_loss = inner_loss[:, -1, :].mean() #last position

        inner_loss.backward(retain_graph=True, create_graph=True)
        nn.utils.clip_grad_norm_(model.base_parameters(), loss_clip)
        opt_inner.step()

        # --- Outer Optimizer ---

        opt_outer.zero_grad()
        data = next(val_dl)
        y = data[:, 1:]
        x = data[:, :-1]

        token_embed = model.token_embed(x)
        logits = model(token_embed, meta=False)
        outer_loss = ce_loss(logits, y)

        outer_loss.backward()
        nn.utils.clip_grad_norm_(model.meta_parameters(), loss_clip)
        opt_outer.step()

        # --- Logging ---

        if step % 50 == 0:
            ppl = torch.exp(outer_loss.detach()).item()
            print(f"{step:>6}  critic={outer_loss.item():.3f}  val-ppl={ppl:6.2f}")


# --- CLI ---
if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",    default="./data")
    ap.add_argument("--batch_size",  type=int, default=4)
    ap.add_argument("--block_size",  type=int, default=2048)
    ap.add_argument("--device",      default="cuda")
    ap.add_argument("--lr_inner",    type=float, default=3e-4)
    ap.add_argument("--lr_outer",    type=float, default=1e-3)
    args = ap.parse_args()

    run(args.data_dir,
        args.batch_size,
        args.block_size,
        args.device,
        args.lr_inner,
        args.lr_outer)
