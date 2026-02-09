import torch
from .manage_datasets.uploadDatasets import get_Card_Dataloader_grouped, get_Willet_Dataloader_grouped
from .model.decoder import *

def load_id2word(path="data/assets/lang/words.txt"):
    id2w = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w, i = line.strip().split()
            id2w[int(i)] = w
    return id2w


def refs_card(batch):
    out = []
    id2w = load_id2word()
    for tr in batch["transcriptions"]:
        if tr is None:
            out.append("")
            continue
        words = []
        for idx in tr.tolist():
            if idx == 0:
                continue
            w = id2w.get(int(idx), "")
            if not w or w in ("<eps>", "<s>", "</s>", "#0"):
                continue
            words.append(w)
        out.append(" ".join(words))
    return out

@torch.no_grad()
def eval_wer(model, decoder, dl, device, get_refs, max_batches=None, log_every=20):
    model.eval()
    refs_all, hyps_all = [], []

    for i, batch in enumerate(dl, start=1):
        x = batch["input_features"].to(device)
        x_lens = batch["n_time_steps"].to(device)
        sess = batch["meta"][0]["session_key"]

        _, _, l3, _ = model(x, x_lens, sess)
        hyps = decode_words(decoder, l3, x_lens)

        refs = get_refs(batch)

        refs_all.extend(refs)
        hyps_all.extend(hyps)

        if (i % log_every) == 0:
            print(f"[WER] batches={i} running={compute_wer(refs_all, hyps_all):.4f}")
        if max_batches is not None and i >= max_batches:
            break

    final = compute_wer(refs_all, hyps_all)
    return final, refs_all, hyps_all

def testWillet(model, decoder, device):
    dl_w_test = get_Willet_Dataloader_grouped(batch_size=16, split="test")
    wer_w, refs_w, hyps_w = eval_wer(
        model, decoder, dl_w_test, device,
        get_refs=lambda b: b["texts"],
        log_every=10,
    )
    print("Willett WER:", wer_w)
    print("REF:", refs_w[0])
    print("HYP:", hyps_w[0])

def testCard(model, decoder, device):
    dl_c_test = get_Card_Dataloader_grouped(batch_size=16, split="test")
    wer_c, refs_c, hyps_c = eval_wer(
        model, decoder, dl_c_test, device,
        get_refs=refs_card,
        log_every=10,
    )
    print("Card WER:", wer_c)
    print("REF:", refs_c[0])
    print("HYP:", hyps_c[0])



