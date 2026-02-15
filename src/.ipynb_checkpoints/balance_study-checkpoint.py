import math
import torch
from pathlib import Path
from tqdm import trange, tqdm
from torch.cuda.amp import autocast, GradScaler
from .manage_datasets.uploadDatasets import *
from .model.encoder import Encoder
from .model.CTC_Loss import Hierarchical_Loss
from .model.decoder import *
from .test import testWillet, testCard, quick_eval_wer
from .model.utils import save_checkpoint, get_device
import random
import os
import csv
import numpy as np


NB_STEPS = 1000
BATCH_SZ = 40

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def concat_ctc_targets(y_pad: torch.Tensor, y_lens: torch.Tensor) -> torch.Tensor:
    parts = []
    for i in range(y_pad.size(0)):
        Li = int(y_lens[i].item())
        if Li > 0:
            parts.append(y_pad[i, :Li])
    return torch.cat(parts, dim=0) if parts else torch.empty((0,), dtype=torch.long, device=y_pad.device)

def lr_warmup_cosine(step: int, total: int, warmup: int, base_lr: float, final_lr: float) -> float:
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    if total <= warmup:
        return final_lr
    t = (step - warmup) / (total - warmup)
    t = max(0.0, min(1.0, t))
    return final_lr + (base_lr - final_lr) * 0.5 * (1.0 + math.cos(math.pi * t))

def train(dataloader, decoder, device, balance, steps=NB_STEPS, base_name="ckpt", save_path="checkpoints"):
    print(f"gpu : {device} | balance : {balance}")

    b = next(iter(dataloader))
    max_id = int(b["seq_class_ids"].max().item())
    PHONEME_CLASSES = max_id + 1
    input_dim = b["input_features"].shape[-1]

    model = Encoder(input_dim=512, d=256, phoneme_class=PHONEME_CLASSES).to(device)

    criterion = Hierarchical_Loss(balance=balance).to(device)

    train_mixed(model, criterion, dataloader, device, steps=steps, log_every=10, 
                decoder=decoder, save_base=base_name, save_folder=save_path)

    return model


def train_mixed(model, criterion, mixed_iter, device, steps=1000, log_every=50, lr=1e-3, 
                decoder=None, save_folder="checkpoints", save_base="ckpt"):
    

    os.makedirs(save_folder, exist_ok=True)

    log_path = os.path.join(save_folder, f"{save_base}_loss.csv")

    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("step,loss\n")

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    pbar = tqdm(range(steps), desc="train")
    running = 0.0
    best_loss = float("inf")

    iterator = iter(mixed_iter)
    current_avg_loss = 0.0

    for step in pbar:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(mixed_iter)
            batch = next(iterator)

        src = batch.get("_source", "?")
        x = batch["input_features"].to(device)
        x_lens = batch["n_time_steps"].to(device)
        y_pad = batch["seq_class_ids"].to(device)
        y_lens = batch["phone_seq_lens"].to(device)
        sess = batch["meta"][0]["session_key"]

        opt.zero_grad(set_to_none=True)

        l1, l2, l3, _ = model(x, x_lens, sess)
        target_1d = concat_ctc_targets(y_pad, y_lens)

        loss = criterion(target=target_1d, l1=l1, l2=l2, l3=l3, input_len=x_lens, target_len=y_lens)

        loss.backward()
        opt.step()

        loss_val = float(loss.detach().item())
        running += loss_val
        
        del l1, l2, l3, loss, x, y_pad

        if (step + 1) % log_every == 0:
            current_avg_loss = running / log_every
            pbar.set_postfix(loss=f"{current_avg_loss:.4f}", src=src)
            running = 0.0

            
            with open(log_path, "a") as f:
                f.write(f"{step + 1},{current_avg_loss:.5f}\n")

        if (step + 1) % 25 == 0:
            if current_avg_loss < best_loss and current_avg_loss > 0:
                best_loss = current_avg_loss
                save_checkpoint(
                    model, opt, step=step + 1,
                    extra={"best_loss": best_loss, "metric": "loss"},
                    folder=save_folder,
                    base_name=save_base
                )
                pbar.write(f"saved best (loss={best_loss:.4f})")

    return model

    
if __name__ == "__main__":

    Path = "./checkpoints/"

    try:
        decoder = build_ta_ctc_decoder(
            tokens=TOKENS,
            lexicon_path="data/assets/lexicon_nostress.txt",
            lm_path="data/assets/3-gram.arpa",
        )
    except Exception as e:
        print(f"[WARN] Decoder off: {e}")
        decoder = None

    device = get_device()
    
    balances_to_test = [0.0,0.3,0.5]
    seeds_to_test = [0,1, 2]

    for balance in balances_to_test:
        for seed in seeds_to_test:
            set_seed(seed)
            
            run_name = f"final_bal{str(balance).replace('.', 'p')}_seed{seed}"
            
            print(f"\n START RUN: {run_name} (Balance={balance}, Seed={seed})")
    
            dl_card = get_Card_Dataloader_grouped(batch_size=BATCH_SZ, split="train", num_workers=2)
            dl_will = get_Willet_Dataloader_grouped(batch_size=BATCH_SZ, split="train", num_workers=2)
            mixed = MixedBatchIterator(loaders={"card": dl_card, "willett": dl_will}, strategy="alternate")
    

            model = train(
                mixed, decoder, device, 
                balance=balance,   
                steps=NB_STEPS, 
                base_name=run_name, 
                save_path=Path
            )

            save_checkpoint(
                model, None, step=NB_STEPS, 
                extra={"seed": seed, "balance": balance}, 
                folder=Path, 
                base_name=f"_{run_name}_FINISHED"
            )
            
            del model
            torch.cuda.empty_cache()
            print(f" End of {run_name}\n")
