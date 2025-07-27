import argparse, json, os, torch, copy
import torch.nn.functional as F
import torch.optim as optim
from model import Transformer, Config
from dataloader import DataLoader
from transformers import AutoTokenizer


def meta_train(model, loader, loss_id, n_steps, lr, tokenizer):

    dev = next(model.parameters()).device
    opt = optim.AdamW(model.parameters(), lr=lr)

    for step in range(n_steps):

        batch = loader.next_batch()          # B × T_tot
        train = batch[:, : loader.T // 4]    # B × L₁
        train_x = train[:, :-1]              # B × L₁-1
        train_y = train[:, 1:]               # B × L₁-1  (shifted)
        val_x = batch[:, loader.T // 4 : -1] # B × L₂-1
        val_y = batch[:, loader.T // 4 + 1:] # B × L₂-1  (shifted)

        # print('--------------------------------')
        # print([tokenizer.decode(x) for x in train_y[0]])

        pred = model(train_x)

        # print('--')
        # print([tokenizer.decode(x) for x in pred[0].argmax(dim=-1)])

        loss_tokens = torch.full(train_y.shape, loss_id, device=dev)
        # Interleave (train_y, pred, loss_tokens) in a 3-way zip fashion
        # For each position, take train_y, then pred, then loss_token, repeat
        # Assume pred is logits, so take argmax to get predicted token ids
        pred_tokens = pred.clone().detach().argmax(dim=-1)
        # Flatten all to 1D for interleaving, then reshape back
        train_y_flat = train_y.reshape(-1)
        pred_flat = pred_tokens.reshape(-1)
        loss_flat = loss_tokens.reshape(-1)
        # Stack and interleave
        stacked = torch.stack([pred_flat, train_y_flat, loss_flat], dim=1)
        interleaved = stacked.view(-1)
        # Reshape to (B, L*3) where L = train_y.size(1)
        B, L = train_y.shape
        interleaved = interleaved.view(B, L * 3)

        # print('--')
        # print([tokenizer.decode(x) for x in interleaved[0]])
        
        # get prediction on interleaved
        pred_interleaved = model(interleaved)
        # print('--')
        # print([tokenizer.decode(x) for x in pred_interleaved[0].argmax(dim=-1)])
    
        #get pred_interleaved only for the indices of the loss tokens
        loss_token_indices = torch.arange(2, interleaved.size(1), 3, device=interleaved.device)
        pred_interleaved_idx = pred_interleaved[:, loss_token_indices, :]
        # print('--')
        # print([tokenizer.decode(x) for x in pred_interleaved_idx[0].argmax(dim=-1)])

        #calulate meta loss
        meta_loss = F.cross_entropy(pred_interleaved_idx.permute(0, 2, 1), train_y)
        
        #calculate lm loss
        lm_loss = F.cross_entropy(pred.permute(0, 2, 1), train_y)

        loss = lm_loss + meta_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # ---------- validation -----------------------------
        with torch.inference_mode():

            pred_val = model(val_x)
            val_loss = F.cross_entropy(pred_val.permute(0, 2, 1), val_y)

        print(f"[meta] {step+1:>4}/{n_steps} lm_loss={lm_loss.item():.3f} "
              f"meta_loss={meta_loss.item():.3f} loss={loss.item():.3f} "
              f"val_loss={val_loss.item():.3f}")

        torch.cuda.empty_cache()


def self_train(model, loader, loss_id, n_steps, lr):

    dev = next(model.parameters()).device
    critic = copy.deepcopy(model).to(dev)
    opt = optim.AdamW(model.parameters(), lr=lr)

    for step in range(n_steps):

        batch  = loader.next_batch()
        train  = batch[:, : loader.T // 4] #1/3 of block size
        train_x = train[:, :-1]
        train_y = train[:, 1:]
        val_x  = batch[:, loader.T // 4 : -1]
        val_y  = batch[:, loader.T // 4 + 1 :]

        # -------- predict forward -------------------------------------
        pred = model(train_x)

        # -------- self-loss forward ------------------------------------
        with torch.inference_mode():

            loss_tokens = torch.full(train_x.shape, loss_id, device=dev)
            pred_tokens = pred.clone().detach().argmax(dim=-1)
            train_y_flat = train_y.reshape(-1)
            pred_flat = pred_tokens.reshape(-1)
            loss_flat = loss_tokens.reshape(-1)
            stacked = torch.stack([pred_flat, train_y_flat, loss_flat], dim=1)
            interleaved = stacked.view(-1)
            B, L = train_y.shape
            interleaved = interleaved.view(B, L * 3)

            pred_meta = critic(interleaved)
            idx = torch.arange(2, interleaved.size(1), 3, device=interleaved.device)
            pred_target = pred_meta[:, idx, :].argmax(dim=-1)
        
        pred_target = pred_target.clone().detach()
        loss = F.cross_entropy(pred.permute(0, 2, 1), pred_target)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.inference_mode():
            pred_val = model(val_x)
            val_loss = F.cross_entropy(pred_val.permute(0, 2, 1), val_y)

        print(f"[self] {step+1:>4}/{n_steps} loss={loss.item():.3f} "
              f"CE={val_loss.item():.3f}")

        del batch, train, train_x, train_y, val_x, val_y, pred, loss_tokens, pred_tokens, train_y_flat, pred_flat, 
        del loss_flat, stacked, interleaved, loss, pred_meta, idx, pred_target, val_loss
        torch.cuda.empty_cache()


def main(cfg):

    device = "cuda"
    assert torch.cuda.is_available()

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
               lr=cfg.lr_meta,
               tokenizer=tok)

    # ------------ Phase 2 ----------------------------------------------------
    self_train(model, loader, loss_id,
                    n_steps=cfg.self_steps,
                    lr=cfg.lr_self)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",    default="./data")
    ap.add_argument("--batch",       type=int,   default=128)
    ap.add_argument("--block",       type=int,   default=128)
    ap.add_argument("--meta_steps",  type=int,   default=1000)
    ap.add_argument("--self_steps",  type=int,   default=1000)
    ap.add_argument("--lr_meta",     type=float, default=1e-4)
    ap.add_argument("--lr_self",     type=float, default=1e-4)
    cfg = ap.parse_args()
    main(cfg)
