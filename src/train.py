import math
import torch
from tqdm import trange, tqdm
from torch.cuda.amp import autocast, GradScaler
from .manage_datasets.uploadDatasets import *
from .model.encoder import Encoder
from .model.CTC_Loss import Hierarchical_Loss
from .model.decoder import *
from .test import testWillet, testCard

NB_STEPS = 350

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

@torch.no_grad()
def sanity_check_session_pure(batch):
    k = batch["meta"][0]["session_key"]
    assert all(m["session_key"] == k for m in batch["meta"])
    return k

def train(dataloader, decoder, device, steps=NB_STEPS):
    print(f"gpu : {device}")

    b = next(iter(dataloader))
    max_id = int(b["seq_class_ids"].max().item())
    PHONEME_CLASSES = max_id + 1
    input_dim = b["input_features"].shape[-1]

    model = Encoder(input_dim=input_dim, d=256, phoneme_class=PHONEME_CLASSES).to(device)
    criterion = Hierarchical_Loss(balance=0.3).to(device)

    train_mixed(model, criterion, dataloader, device, steps=steps, log_every=50, decoder=decoder)

    model.eval()
    b = next(iter(dataloader))
    x = b["input_features"].to(device)
    x_lens = b["n_time_steps"].to(device)
    session_key = b["meta"][0]["session_key"]
    l1, l2, l3, _ = model(x, x_lens, session_key)
    pred = ta_decode_batch(decoder, l3, x_lens)

    return model


def concat_ctc_targets(y_pad, y_lens):
    parts = []
    for i in range(y_pad.size(0)):
        Li = int(y_lens[i].item())
        if Li > 0:
            parts.append(y_pad[i, :Li])
    return torch.cat(parts, dim=0) if parts else torch.empty((0,), dtype=torch.long, device=y_pad.device)


def train_mixed(model, criterion, mixed_iter, device, steps=1000, log_every=50, lr=1e-3, decoder=None):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    pbar = tqdm(range(steps), desc="train")
    running = 0.0

    for step in pbar:
        batch = next(iter(mixed_iter)) 

        x = batch["input_features"].to(device)
        x_lens = batch["n_time_steps"].to(device)
        y_pad = batch["seq_class_ids"].to(device)
        y_lens = batch["phone_seq_lens"].to(device)

        sess = batch["meta"][0]["session_key"]

        opt.zero_grad(set_to_none=True)

        l1, l2, l3, _ = model(x, x_lens, sess)

        target_1d = concat_ctc_targets(y_pad, y_lens)

        loss = criterion(target=target_1d, l1=l1 ,l2=l2 ,l3=l3 ,input_len=x_lens, target_len=y_lens)

        loss.backward()
        opt.step()

        loss_val = float(loss.detach().item())
        running += loss_val

        if (step + 1) % log_every == 0:
            avg = running / log_every
            src = batch.get("_source", "?")
            pbar.set_postfix(loss=f"{avg:.4f}", src=src)
            print("")
            running = 0.0

            if decoder is not None:
                model.eval()
                with torch.no_grad():
                    pred_texts = ta_decode_batch(decoder, l3, x_lens)
                model.train()

    return model


if __name__ == "__main__":
    
    dl_card = get_Card_Dataloader_grouped(batch_size=16, split="train")
    dl_will = get_Willet_Dataloader_grouped(batch_size=16, split="train")

    mixed = MixedBatchIterator(loaders={"card": dl_card, "willett": dl_will}, strategy="alternate")

    decoder = build_ta_ctc_decoder(
        lexicon_path="data/assets/lexicon_nostress.txt",
        arpa_path="data/assets/3-gram.arpa",
        tokens=TOKENS,
        lm_weight=2.0,
        word_score=1.0,
        beam_size=100,
    )

    device = get_device()

    model = train(mixed, decoder, device)

    testCard(model, decoder, device)
