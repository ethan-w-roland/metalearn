import os, argparse, itertools, torch, torch.nn as nn, torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from dataloader import create_loader
# --------------------------------------------------------------------------- #
# 1.  CLI arguments                                                           #
# --------------------------------------------------------------------------- #
ap = argparse.ArgumentParser()
ap.add_argument("--model_name", default="meta-llama/Llama-3.2-1B")
ap.add_argument("--data_dir",   default="./data")
ap.add_argument("--out_dir",    default="./output")
ap.add_argument("--batch_size", type=int, default=4)
ap.add_argument("--block_size", type=int, default=2048)
ap.add_argument("--warm_steps", type=int, default=500)
ap.add_argument("--total_steps", type=int, default=10000)
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
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

train_data = create_loader(os.path.join(args.data_dir, "train.bin"),
                         args.block_size, args.batch_size, device=args.device)
val_data = create_loader(os.path.join(args.data_dir, "val.bin"),
                         args.block_size, args.batch_size, device=args.device)

# --------------------------------------------------------------------------- #
# 3.  Load LLAMA + inject LoRA                                                #
# --------------------------------------------------------------------------- #
base = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch.bfloat16,      # or torch.float16
    device_map="auto",               # places weights on GPU
)
lora_cfg = LoraConfig(
    r=16, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj",
                                         "gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(base, lora_cfg)               # PEFT wraps the base model
model.print_trainable_parameters()
model.to(device)

# helper to toggle all LoRA layers ------------------------------------------
def set_lora(model, flag: bool):
    for m in model.modules():
        if hasattr(m, "disable_adapters"):
            m.disable_adapters(not flag)

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
opt_inner = optim.AdamW(base_iter(model), lr=args.inner_lr)
opt_outer = optim.AdamW(lora_iter(model), lr=args.outer_lr)
ce_loss   = nn.CrossEntropyLoss()

# snapshot of θ₀ for ML³ warm-up
theta0 = {k: p.detach().clone() for k, p in model.state_dict().items()
          if "lora_" not in k}

def reset_base():
    with torch.no_grad():
        for k, v in theta0.items():
            model.state_dict()[k].copy_(v)

for i in range(args.total_steps):

    if i < args.warm_steps:
        reset_base()

    # -------------- inner θ step ----------------------------------
    opt_inner.zero_grad()
    train_batch = next(train_data)
    set_lora(model, False)
    logits = model(train_batch).logits
    pred = logits.argmax(-1)
    meta_inp = torch.cat([train_batch, pred], dim=1)
    set_lora(model, True)
    inner_loss = model(meta_inp).mean()        # forward in learned-loss mode
    inner_loss.backward(retain_graph=True, create_graph=True)
    opt_inner.step()

    # -------------- outer ψ step ---------------------------------
    set_lora(model, False)
    val_batch = next(val_data)
    x, y = val_batch[:, :-1], val_batch[:, 1:]
    val_logits = model(x).logits
    outer_loss = ce_loss(val_logits.view(-1, tokenizer.vocab_size), y.reshape(-1))
    outer_loss.backward()
    opt_outer.step()

    if (i+1) % 100 == 0:
        print(f"{i+1}/{args.total_steps} CE={outer_loss.item():.3f}")