import os, argparse, torch, higher
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from dataloader import DataLoader
from pathlib import Path
from dotenv import load_dotenv
from torch.amp.autocast_mode import autocast

load_dotenv()

# --------------------------------------------------------------------------- #
# 1.  CLI arguments                                                           #
# --------------------------------------------------------------------------- #

cur_dir = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--model_name", default="meta-llama/Llama-3.2-1B")
ap.add_argument("--data_dir",   default=cur_dir / "data")
ap.add_argument("--out_dir",    default=cur_dir / "output")
ap.add_argument("--batch_size", type=int, default=2)
ap.add_argument("--block_size", type=int, default=256)
ap.add_argument("--warm_steps", type=int, default=500)
ap.add_argument("--inner_lr",   type=float, default=2e-5)
ap.add_argument("--outer_lr",   type=float, default=1e-4)
ap.add_argument("--device",     default="cuda")
args = ap.parse_args()

device = torch.device(args.device)
if device.type == "cuda":
    assert torch.cuda.is_available(), "CUDA is not available"

# --------------------------------------------------------------------------- #
# 2.  Load tokenizer & dataset                                                #
# --------------------------------------------------------------------------- #

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name, use_fast=True, token=os.getenv("HF_TOKEN"))

train_data = DataLoader(
    args.data_dir / "train.bin",
    B=args.batch_size,
    T=args.block_size,
    device=args.device,
)

val_data = DataLoader(
    args.data_dir / "test.bin",
    B=args.batch_size,
    T=args.block_size,
    device=args.device,
)

# --------------------------------------------------------------------------- #
# 3.  Load LLAMA + inject LoRA                                                #
# --------------------------------------------------------------------------- #

base = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch.bfloat16,
)

lora_cfg = LoraConfig(
    r=16, 
    lora_alpha=16, 
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"],
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

model = get_peft_model(base, lora_cfg) # PEFT wraps the base model
model.gradient_checkpointing_enable()
model.config.use_cache = False
model.print_trainable_parameters()
model.to(device)

def base_iter(model):
    for n, p in model.named_parameters():
        if "lora_" not in n:
            yield p

def lora_iter(model):
    for n, p in model.named_parameters():
        if "lora_" in n:
            yield p

# --------------------------------------------------------------------------- #
# 4.  Optimisers                                                              #
# --------------------------------------------------------------------------- #

opt_inner = None
opt_outer = optim.AdamW(lora_iter(model), lr=args.outer_lr)
ce_loss   = nn.CrossEntropyLoss()

# snapshot of θ₀ for warm-up
start_weights = {k: p.detach().clone() for k, p in model.state_dict().items() if "lora_" not in k}

def reset_base():
    with torch.no_grad():
        for k, v in start_weights.items():
            model.state_dict()[k].copy_(v)

# --------------------------------------------------------------------------- #
# 5.  Training loop (ML³ warm-up  → AdaLFL online)                            #
# --------------------------------------------------------------------------- #

with autocast(dtype=torch.bfloat16, device_type="cuda"):

    for step in range(len(train_data)):

        if step < args.warm_steps:
            reset_base()

        # -----------------------------------------------------------
        # build a FRESH inner optimiser & higher context each step
        # -----------------------------------------------------------
        opt_inner = optim.SGD(base_iter(model), lr=args.inner_lr)
        with higher.innerloop_ctx(
                model, opt_inner,
                copy_initial_weights=False,
                track_higher_grads=True) as (fmodel, diffopt):

            # ---------------- inner θ update (LoRA ON) ------------
            batch = train_data.next_batch()             # [B,T+1] on GPU
            x, y = batch[:, :-1], batch[:, 1:]

            with fmodel.disable_adapter():              # LoRA OFF
                logits = fmodel(x).logits
            y_hat = logits.argmax(-1)

            critic_in = torch.cat([x, y_hat, y], dim=1)
            inner_loss = fmodel(critic_in).logits[:, -1, :].mean()    # LoRA ON
            diffopt.step(inner_loss)                        # differentiable

            # ---------------- outer ψ update (LoRA OFF) ----------
            val_batch = val_data.next_batch()
            xv, yv = val_batch[:, :-1], val_batch[:, 1:]

            with fmodel.disable_adapter():
                val_logits = fmodel(xv).logits               # uses θ′
            outer_loss = ce_loss(val_logits.permute(0, 2, 1), yv)

            opt_outer.zero_grad()
            outer_loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_iter(model), 1.0)
            opt_outer.step()

        # ------------- logging -----------------------------------
        if (step + 1) % 100 == 0:
            print(f"{step+1}/{len(train_data)}  CE={outer_loss.item():.3f}")