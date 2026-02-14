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

NB_STEPS = 1000
BATCH_SZ = 40

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

def train(dataloader, decoder, device, steps=NB_STEPS):
    print(f"gpu : {device}")

    b = next(iter(dataloader))
    max_id = int(b["seq_class_ids"].max().item())
    PHONEME_CLASSES = max_id + 1
    input_dim = b["input_features"].shape[-1]

    model = Encoder(input_dim=512, d=256, phoneme_class=PHONEME_CLASSES,learnable=False).to(device)
    criterion = Hierarchical_Loss(balance=0.3).to(device)

    train_mixed(model, criterion, dataloader, device, steps=steps, log_every=10, decoder=decoder)

    return model


def train_mixed(model, criterion, mixed_iter, device, steps=1000, log_every=50, lr=1e-3, decoder=None, save_folder="checkpoints", save_base="ckpt"):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    pbar = tqdm(range(steps), desc="train")
    running = 0.0
    best_wer = float("inf")
    best_loss = float("inf")
    iterator = iter(mixed_iter)
    current_avg_loss = 0.0

    for step in pbar:

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

        if (step + 1) % log_every == 0:
            # Display loss
            current_avg_loss = running / log_every
            src = batch.get("_source", "?")
            pbar.set_postfix(loss=f"{current_avg_loss:.4f}", src=src)
            print("")
            running = 0.0

            if decoder is not None:
                model.eval()
                with torch.no_grad():
                    pred_texts = ta_decode_batch(decoder, l3, x_lens)
                model.train()

        if (step + 1) % 25 == 0:
            # Save checkpoint
            if current_avg_loss < best_loss:
                best_loss = current_avg_loss
                save_checkpoint(
                    model, opt, step=step + 1,
                    extra={"best_loss": best_loss, "metric": "loss"},
                    folder=save_folder,
                    base_name=save_base
                )
                pbar.write(f"✅ saved best (loss={best_loss:.4f})")

    return model


if __name__ == "__main__":
    dl_card = get_Card_Dataloader_grouped(batch_size=BATCH_SZ, split="train", num_workers=4)
    dl_will = get_Willet_Dataloader_grouped(batch_size=BATCH_SZ, split="train", num_workers=4)

    mixed = MixedBatchIterator(loaders={"card": dl_card, "willett": dl_will}, strategy="alternate")

    try:
        decoder = build_ta_ctc_decoder(
            tokens=TOKENS,
            lexicon_path="data/assets/lexicon_nostress.txt",
            lm_path="data/assets/3-gram.arpa",
        )
    except Exception as e:
        print(f"[WARN] Decoder désactivé (flashlight/kenlm manquant): {e}")
        decoder = None

    device = get_device()

    model = train(mixed, decoder, device, steps=NB_STEPS)

    # Checkpoint automatique
    save_checkpoint(model, None, step=NB_STEPS, extra={"test": "checkpoint"})

    for k, layer in model.session_align.transforms.items():
        W = layer.weight.detach()
        print(k, torch.norm(W - torch.eye(W.size(0), device=W.device)))

    del model
    torch.cuda.empty_cache()
    # checkpoint manuel
    # save_checkpoint(model, None, step=5, extra={"note": "manual"}, base_name="_manual_best")
