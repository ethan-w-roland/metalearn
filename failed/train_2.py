# train_self_loss.py ---------------------------------------------------------
import argparse, json, os, torch, higher
from copy import deepcopy
import torch.nn.functional as F
import torch.optim as optim
from model import Transformer, Config
from dataloader import DataLoader
from transformers import AutoTokenizer

# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def make_mask(tokens: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Segment-aware causal attention mask (B,1,T,T)."""
    _, T = tokens.shape
    dev   = tokens.device
    eos   = tokens.eq(eos_id)
    seg   = torch.cumsum(eos.int(), 1) - eos.int()          # segment IDs
    same  = seg.unsqueeze(2).eq(seg.unsqueeze(1))           # same segment
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=dev))
    return (same & causal).unsqueeze(1)

@torch.no_grad()
def ema_merge(dst: torch.nn.Module, src: torch.nn.Module, tau: float = 0.5):
    """θ_dst ← (1−τ) θ_dst + τ θ_src   (in-place)."""
    for p_d, p_s in zip(dst.parameters(), src.parameters()):
        p_d.mul_(1 - tau).add_(p_s.detach(), alpha=tau)
    for b_d, b_s in zip(dst.buffers(), src.buffers()):
        if b_d.dtype.is_floating_point:
            b_d.mul_(1 - tau).add_(b_s.detach(), alpha=tau)

# --------------------------------------------------------------------------- #
#  phase 1 : full AdaLFL (meta-training)                                      #
# --------------------------------------------------------------------------- #
def meta_train(model, loader, loss_token_id, eos_id,
               n_steps, lr_inner, lr_outer, clip):

    dev = next(model.parameters()).device
    opt_outer = optim.AdamW(model.parameters(), lr=lr_outer)

    for step in range(n_steps):

        batch = loader.next_batch()          # (B, T_tot)
        train = batch[:, : loader.T // 3]    # B × L₁
        val_x = batch[:, loader.T // 3 : -1] # B × L₂-1
        val_y = batch[:, loader.T // 3 + 1:] # B × L₂-1  (shifted)

        opt_inner  = optim.SGD(model.parameters(), lr=lr_inner)

        with higher.innerloop_ctx(model, opt_inner,
                                  copy_initial_weights=False,
                                  track_higher_grads=True) as (fmodel, diffopt):

            z_in  = fmodel.embed_in(train)                       # (B,L₁,D)
            m_in  = make_mask(train[:, :-1], eos_id)
            z_out = fmodel(z_in[:, :-1], mask=m_in)                   # (B,L₁-1,D)
            z_loss = fmodel.embed_in(torch.full((batch.size(0), 1),
                                                loss_token_id, device=dev))
            meta_inp = torch.cat([z_in, z_out, z_loss], 1)            # (B,⋯,D)

            inner_loss = fmodel(meta_inp)[:, -1].mean()  # scalar
            diffopt.step(inner_loss)                    # differentiable θ′

            # ---------- outer (real CE) step -----------------------------
            m_out   = make_mask(val_x, eos_id)
            e_out   = fmodel.embed_in(val_x)              # tokens ➜ embed
            h_out   = fmodel(e_out, mask=m_out)             # encoder
            logits  = fmodel.embed_out(h_out)               # ➜ vocab
            ce      = F.cross_entropy(logits.permute(0, 2, 1), val_y)

            ce.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt_outer.step()
            opt_outer.zero_grad(set_to_none=True)
            ema_merge(model, fmodel, tau=0.5)

        print(f"[meta] {step+1:>4}/{n_steps} pred-loss={inner_loss.item():.3f} "
              f"CE={ce.item():.3f}")

        del fmodel, diffopt, opt_inner
        torch.cuda.empty_cache()

# --------------------------------------------------------------------------- #
#  phase 2 : self-loss-only fine-tuning                                       #
# --------------------------------------------------------------------------- #
def self_loss_train(model, loader,
                    loss_token_id, eos_id,
                    n_steps, lr_inner, clip):

    dev = next(model.parameters()).device
    opt = optim.AdamW(model.parameters(), lr=lr_inner)
    critic = deepcopy(model).to(dev)
    critic.eval()                       # we never update critic itself
    for p in critic.parameters():       # disable grad tracking for critic weights
        p.requires_grad_(False)

    for step in range(n_steps):

        batch  = loader.next_batch()
        train  = batch[:, : loader.T // 3]
        val_x  = batch[:, loader.T // 3 : -1]
        val_y  = batch[:, loader.T // 3 + 1 :]

        # -------- self-loss forward -------------------------------------
        z_in  = model.embed_in(train)
        m_in  = make_mask(train[:, :-1], eos_id)
        z_out = model(z_in[:, :-1], mask=m_in)
        z_L   = model.embed_in(torch.full((batch.size(0), 1),
                                          loss_token_id, device=dev))
        loss_inp = torch.cat([z_in, z_out, z_L], 1)
        # Forward through the frozen critic while keeping the graph for loss_inp
        pred_loss = critic(loss_inp)[:, -1].mean()

        opt.zero_grad()
        pred_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()

        with torch.inference_mode():                            # CE probe
            mv  = make_mask(val_x, eos_id)
            ev  = model.embed_in(val_x)
            hv  = model(ev, mask=mv)
            logits = model.embed_out(hv)
            ce_val = F.cross_entropy(logits.permute(0, 2, 1), val_y)

        print(f"[self] {step+1:>4}/{n_steps} pred-loss={pred_loss.item():.3f} "
              f"CE={ce_val.item():.3f}")

# --------------------------------------------------------------------------- #
#  main                                                                       #
# --------------------------------------------------------------------------- #
def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
    tok.add_special_tokens({"additional_special_tokens": ["[LOSS]"]})
    loss_id = tok.convert_tokens_to_ids("[LOSS]")

    with open(os.path.join(cfg.data_dir, "metadata.json")) as f:
        V = json.load(f)["all"]["vocab_size"] + 1

    model = Transformer(Config(vocab_size=V, block_size=cfg.block)).to(device)

    loader = DataLoader(
        filename=f"{cfg.data_dir}/train.bin",
        B=cfg.batch,
        T=cfg.block + cfg.block // 2,
        device=device)

    # ------------ Phase 1 ----------------------------------------------------
    meta_train(model, loader, loss_id, tok.eos_token_id,
               n_steps=cfg.meta_steps,
               lr_inner=cfg.lr_inner,
               lr_outer=cfg.lr_outer,
               clip=1.0)

    # ------------ Phase 2 ----------------------------------------------------
    self_loss_train(model, loader, loss_id, tok.eos_token_id,
                    n_steps=cfg.self_steps,
                    lr_inner=cfg.lr_inner,
                    clip=1.0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",      default="./data")
    ap.add_argument("--batch",         type=int,   default=8)
    ap.add_argument("--block",         type=int,   default=256)
    ap.add_argument("--meta_steps",    type=int,   default=500)
    ap.add_argument("--self_steps",    type=int,   default=500)
    ap.add_argument("--lr_inner",      type=float, default=1e-3)
    ap.add_argument("--lr_outer",      type=float, default=1e-3)
    cfg = ap.parse_args()
    main(cfg)
