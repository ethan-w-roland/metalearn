"""
@author: ethan-w-roland
@date: 2025-07-06
@title: Shared Backbone Meta-Learned Loss Function
"""

import argparse, json, os, torch, higher
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import Transformer, Config
from dataloader import DataLoader
from transformers import AutoTokenizer

def make_mask(tokens: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    _, seq_len = tokens.size()
    device = tokens.device
    eos_mask = tokens == eos_token_id
    seg_ids = torch.cumsum(eos_mask.int(), dim=1) - eos_mask.int()
    same_segment = seg_ids.unsqueeze(2).eq(seg_ids.unsqueeze(1))
    causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
    allowed = same_segment & causal
    return allowed.unsqueeze(1)

@torch.no_grad()
def interpolate_models(model_outer: Transformer,
                       model_inner: Transformer, tau: float = 0.5) -> None:
    """
    In-place EMA: θ_out ← (1−τ) θ_out  +  τ θ_in   with  θ_in DETACHED.
    """
    for p_out, p_in in zip(model_outer.parameters(), model_inner.parameters()):
        p_out.mul_(1.0 - tau).add_(p_in.detach(), alpha=tau)

    for b_out, b_in in zip(model_outer.buffers(), model_inner.buffers()):
        if b_out.dtype.is_floating_point:
            b_out.mul_(1.0 - tau).add_(b_in.detach(), alpha=tau)

def load_vocab(token_dir: str) -> int:
    with open(os.path.join(token_dir, "metadata.json")) as f:
        return json.load(f)["all"]["vocab_size"]

def run(
        data_dir: str,
        block_size: int,
        batch_size: int,
        lr_inner: float,
        lr_outer: float,
) -> None:

    assert torch.cuda.is_available()
    device = "cuda"
    
    # --- Model & Loaders ---

    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
    vocab_size = load_vocab(data_dir)
    vocab_size += 1 # add [LOSS] token
    config = Config(vocab_size=vocab_size, block_size=block_size)
    model = Transformer(config).to(device)

    special_tokens_dict = {"additional_special_tokens": ["[LOSS]"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    loss_token_id = tokenizer.convert_tokens_to_ids("[LOSS]")
    eos_token_id = tokenizer.eos_token_id

    loader = DataLoader(
        filename=f"{data_dir}/train.bin",
        B=batch_size,
        T=block_size + block_size // 2,
        device=device)

    # --- Optimizers ---

    opt_outer = optim.AdamW(model.parameters(), lr=lr_outer)
    loss_clip = 1.0

    for step in range(len(loader)):

        # --- Gather Data ---

        data = loader.next_batch()
        inner_data = data[:, :block_size // 2] #first 1/3
        outer_data = data[:, block_size // 2:] #last 2/3
        outer_data_x = outer_data[:, :-1]
        outer_data_y = outer_data[:, 1:]

        # --- Initialize Inner Optimizer ---

        opt_inner = optim.SGD(model.parameters(), lr=lr_inner)

        with higher.innerloop_ctx(
                model, opt_inner,
                copy_initial_weights=False,
                track_higher_grads=True) as (inner_model, diff_opt_inner):

            # --- Inner Optimizer ---

            embed = inner_model.embed_in(inner_data)
            mask = make_mask(inner_data[:, :-1], eos_token_id)
            embed_out = inner_model(embed[:, :-1], mask=mask)
            loss_embed = inner_model.embed_in(torch.tensor([[loss_token_id]], device=device))
            loss_embed = loss_embed.repeat(batch_size, 1, 1) #repeat over batch

            meta_inp = torch.cat([embed, embed_out, loss_embed], dim=1)
            inner_loss = model(meta_inp)
            inner_loss = inner_loss[:, -1, :].mean() #last position
            diff_opt_inner.step(inner_loss)

            # --- Outer Optimizer ---

            mask = make_mask(outer_data_x, eos_token_id)
            embed = inner_model.embed_in(outer_data_x)
            embed_out = inner_model(embed, mask=mask)
            logits = inner_model.embed_out(embed_out)

            outer_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                outer_data_y.reshape(-1),
                reduction="mean",
            )

            outer_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), loss_clip)
            opt_outer.step()
            opt_outer.zero_grad(set_to_none=True)

            # --- Interpolate Models ---

            interpolate_models(model, inner_model)
            
            # --- Logging ---

            ppl = torch.exp(outer_loss.detach()).item()
            print(f"{step:>6}  critic={outer_loss.item():.3f}  val-ppl={ppl:6.2f}")

        del inner_model, diff_opt_inner, opt_inner
        torch.cuda.empty_cache()


# --- CLI ---
if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",    default="./data")
    ap.add_argument("--batch_size",  type=int, default=64)
    ap.add_argument("--block_size",  type=int, default=256) #context length
    ap.add_argument("--lr_inner",    type=float, default=3e-4)
    ap.add_argument("--lr_outer",    type=float, default=1e-3)
    args = ap.parse_args()

    run(
        data_dir=args.data_dir,
        block_size=args.block_size,
        batch_size=args.batch_size,
        lr_inner=args.lr_inner,
        lr_outer=args.lr_outer)
