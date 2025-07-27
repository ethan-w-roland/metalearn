import argparse, json, os, torch, higher
from copy import deepcopy
import torch.nn.functional as F
import torch.optim as optim
from model_2 import Transformer, Config
from dataloader import DataLoader
from transformers import AutoTokenizer


# --------------------------------------------------------------------------- #
#  phase 1 : full AdaLFL (meta-training)                                      #
# --------------------------------------------------------------------------- #
def meta_train(model, loader, loss_token_id, n_steps, lr_inner, lr_outer, tokenizer):

    dev = next(model.parameters()).device
    opt_outer = optim.AdamW(model.parameters(), lr=lr_outer)

    for step in range(n_steps):

        batch = loader.next_batch()          # B × T_tot
        train = batch[:, : loader.T // 4]    # B × L₁
        val_x = batch[:, loader.T // 4 : -1] # B × L₂-1
        val_y = batch[:, loader.T // 4 + 1:] # B × L₂-1  (shifted)

        opt_inner  = optim.SGD(model.parameters(), lr=lr_inner)

        with higher.innerloop_ctx(model, opt_inner,
                                  copy_initial_weights=False,
                                  track_higher_grads=True) as (fmodel, diffopt):

            print('--------------------------------')
            print(tokenizer.decode(train[0]))

            train_x = train[:, :-1]
            mask = fmodel.make_mask(train_x)
            train_x_embed = fmodel.inp_emb(train_x)
            pred_embed = fmodel(train_x_embed, mask)

            pred_logits = fmodel.out_emb(pred_embed)
            print(tokenizer.decode(pred_logits[0].argmax(dim=-1)))

            loss_tokens = torch.full(train_x.shape, loss_token_id, device=dev)
            loss_tokens_embed = fmodel.inp_emb(loss_tokens)

            # print(f"batch.shape: {batch.shape}")
            # print(f"train.shape: {train.shape}")
            # print(f"train_x.shape: {train_x.shape}")
            # print(f"train_x_embed.shape: {train_x_embed.shape}")
            # print(f"pred_embed.shape: {pred_embed.shape}")
            # print(f"loss_tokens_embed.shape: {loss_tokens_embed.shape}")

            meta_inp = torch.cat([train_x_embed, pred_embed, loss_tokens_embed], 1)
            target_embed = model(meta_inp)[:, -loss_tokens.size(1):, :]

            target_logits = fmodel.out_emb(target_embed)
            print(tokenizer.decode(target_logits[0].argmax(dim=-1)))

            # print(f"target_embed.shape: {target_embed.shape}")

            inner_loss = F.mse_loss(target_embed, pred_embed)
            diffopt.step(inner_loss)

            # ---------- outer (real CE) step -----------------------------
            mask = fmodel.make_mask(val_x)
            val_x_embed = fmodel.inp_emb(val_x)
            pred_embed = fmodel(val_x_embed, mask)
            pred_logits = fmodel.out_emb(pred_embed)
            
            # print(f"pred_logits.shape: {pred_logits.shape}")
            # print(f"val_y.shape: {val_y.shape}")

            outer_loss = F.cross_entropy(pred_logits.permute(0, 2, 1), val_y)
            outer_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            opt_outer.step()
            opt_outer.zero_grad(set_to_none=True)

        print(f"[meta] {step+1:>4}/{n_steps} loss={inner_loss.item():.3f} "
              f"CE={outer_loss.item():.3f}")

        del fmodel, diffopt, opt_inner
        torch.cuda.empty_cache()

# --------------------------------------------------------------------------- #
#  phase 2 : self-loss-only fine-tuning                                       #
# --------------------------------------------------------------------------- #
def self_loss_train(model, loader, loss_token_id, n_steps, lr):

    dev = next(model.parameters()).device
    opt = optim.AdamW(model.parameters(), lr=lr)

    for step in range(n_steps):

        batch  = loader.next_batch()
        train  = batch[:, : loader.T // 4] #1/3 of block size
        val_x  = batch[:, loader.T // 4 : -1]
        val_y  = batch[:, loader.T // 4 + 1 :]

        # -------- self-loss forward -------------------------------------
        train_x = train[:, :-1]
        mask = model.make_mask(train_x)
        train_x_embed = model.inp_emb(train_x)
        pred_embed = model(train_x_embed, mask)

        loss_tokens = torch.full(train_x.shape, loss_token_id, device=dev)
        loss_tokens_embed = model.inp_emb(loss_tokens)

        meta_inp = torch.cat([train_x_embed, pred_embed, loss_tokens_embed], 1)
        target_embed = model(meta_inp)[:, -loss_tokens.size(1):, :]
        loss = F.mse_loss(target_embed, pred_embed)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.inference_mode():
            mask = model.make_mask(val_x)
            val_x_embed = model.inp_emb(val_x)
            pred_embed = model(val_x_embed, mask)
            pred_logits = model.out_emb(pred_embed)
            val_loss = F.cross_entropy(pred_logits.permute(0, 2, 1), val_y)

        print(f"[self] {step+1:>4}/{n_steps} loss={loss.item():.3f} "
              f"CE={val_loss.item():.3f}")

# --------------------------------------------------------------------------- #
#  main                                                                       #
# --------------------------------------------------------------------------- #
def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-1.25M")
    tok.add_special_tokens({"additional_special_tokens": ["[LOSS]"]})
    loss_id = tok.convert_tokens_to_ids("[LOSS]")

    with open(os.path.join(cfg.data_dir, "metadata.json")) as f:
        V = json.load(f)["all"]["vocab_size"] + 1 # +1 for the loss token
    eos_id = tok.eos_token_id

    config = Config(vocab_size=V, block_size=cfg.block, eos_id=eos_id)
    model = Transformer(config).to(device)

    loader = DataLoader(
        filename=f"{cfg.data_dir}/train.bin",
        B=cfg.batch,
        T=(cfg.block//3) + cfg.block,
        device=device)

    # ------------ Phase 1 ----------------------------------------------------
    meta_train(model, loader, loss_id,
               n_steps=cfg.meta_steps,
               lr_inner=cfg.lr_inner,
               lr_outer=cfg.lr_outer,
               tokenizer=tok)

    # ------------ Phase 2 ----------------------------------------------------
    self_loss_train(model, loader, loss_id,
                    n_steps=cfg.self_steps,
                    lr=cfg.lr_self)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",    default="./data")
    ap.add_argument("--batch",       type=int,   default=8)
    ap.add_argument("--block",       type=int,   default=256)
    ap.add_argument("--meta_steps",  type=int,   default=500)
    ap.add_argument("--self_steps",  type=int,   default=500)
    ap.add_argument("--lr_inner",    type=float, default=1e-3)
    ap.add_argument("--lr_outer",    type=float, default=1e-3)
    ap.add_argument("--lr_self",     type=float, default=1e-3)
    cfg = ap.parse_args()
    main(cfg)
