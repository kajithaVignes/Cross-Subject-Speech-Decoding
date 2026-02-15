import torch
from .manage_datasets.uploadDatasets import get_Card_Dataloader_grouped, get_Willet_Dataloader_grouped
from .model.decoder import *
from .model.utils import load_checkpoint, get_device
from .model.encoder import Encoder
from .model.CTC_Loss import Hierarchical_Loss

def refs_card(batch):
    """
    Extrait les références textuelles du batch Card.
    """
    texts = batch.get("texts", [])

    if all(t is None for t in texts):
        print(f"All texts are None in this batch!")
        return [""] * len(texts)
    
    return [t if t is not None else "" for t in texts]

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

def quick_eval_wer(model, decoder, val_iter, device, max_batches=20):
    model.eval()
    refs, hyps = [], []
    it = iter(val_iter)
    
    for _ in range(max_batches):
        b = next(it)
        x = b["input_features"].to(device)
        x_lens = b["n_time_steps"].to(device)
        sess = b["meta"][0]["session_key"]
        
        _, _, l3, _ = model(x, x_lens, sess)
        pred = ta_decode_batch(decoder, l3.float(), x_lens)
        
        refs.extend(b["texts"])
        hyps.extend(pred)
    
    return compute_wer(refs, hyps)

def testWillet(model, decoder, device):
    dl_w_test = get_Willet_Dataloader_grouped(batch_size=16, split="train")

    wer_w, refs_w, hyps_w = eval_wer(
        model, decoder, dl_w_test, device,
        get_refs=lambda b: b["texts"],
        log_every=10,
        max_batches=3
    )
    print("\n" + "="*60)
    print("EXEMPLES DE PRÉDICTIONS:")
    print("="*60)
    for i in range(min(10, len(refs_w))):
        ref_words = refs_w[i].split() if refs_w[i] else []
        hyp_words = hyps_w[i].split() if hyps_w[i] else []
        print(f"[{i}]")
        print(f"  REF ({len(ref_words)} mots): {refs_w[i][:100]}")
        print(f"  HYP ({len(hyp_words)} mots): {hyps_w[i][:100]}")
    print("="*60 + "\n")
    
    print(f"Willett WER: {wer_w:.4f} ({wer_w * 100:.2f}%)")
    if len(refs_w) > 0:
        print("REF:", refs_w[0])
        print("HYP:", hyps_w[0])

def testCard(model, decoder, device, batch_size=16):
    dl_c_test = get_Card_Dataloader_grouped(batch_size=16, split="val")

    wer_c, refs_c, hyps_c = eval_wer(
        model, decoder, dl_c_test, device,
        get_refs=refs_card,
        log_every=10,
        max_batches=3,
    )
    
    print("\n" + "="*60)
    print("EXEMPLES DE PRÉDICTIONS:")
    print("="*60)
    for i in range(min(10, len(refs_c))):
        ref_words = refs_c[i].split() if refs_c[i] else []
        hyp_words = hyps_c[i].split() if hyps_c[i] else []
        print(f"[{i}]")
        print(f"  REF ({len(ref_words)} mots): {refs_c[i][:100]}")
        print(f"  HYP ({len(hyp_words)} mots): {hyps_c[i][:100]}")
    print("="*60 + "\n")
    
    print(f"Card WER: {wer_c:.4f} ({wer_c * 100:.2f}%)")
    if len(refs_c) > 0:
        print("REF:", refs_c[0])
        print("HYP:", hyps_c[0])


def ctc_greedy_decode_ids(logits, lengths, blank_id=0):
    """
    logits: (B, T, C)
    lengths: (B,)
    return: list[list[int]] prédiction par batch, après CTC collapse
    """
    pred = logits.argmax(dim=-1)  # (B,T)
    out = []
    B = pred.size(0)
    for b in range(B):
        T = int(lengths[b].item())
        seq = pred[b, :T].tolist()
        collapsed = []
        prev = None
        for x in seq:
            if x == prev:
                continue
            prev = x
            if x != blank_id:
                collapsed.append(x)
        out.append(collapsed)
    return out

def get_ref_ids_from_batch(batch):
    y_pad = batch["seq_class_ids"]      # (B, Lmax)
    y_lens = batch["phone_seq_lens"]    # (B,)
    refs = []
    for i in range(y_pad.size(0)):
        Li = int(y_lens[i].item())
        refs.append(y_pad[i, :Li].tolist())
    return refs

def edit_distance(a, b):
    # Levenshtein classique
    n, m = len(a), len(b)
    dp = list(range(m+1))
    for i in range(1, n+1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m+1):
            cur = dp[j]
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j] + 1,        
                        dp[j-1] + 1,     
                        prev + cost)     
            prev = cur
    return dp[m]

def per_score(ref_seqs, hyp_seqs):
    tot_edits, tot_len = 0, 0
    for r, h in zip(ref_seqs, hyp_seqs):
        tot_edits += edit_distance(r, h)
        tot_len += max(1, len(r))
    return tot_edits / tot_len


@torch.no_grad()
def eval_per_no_decoder(model, dl, device, max_batches=50, blank_id=0, log_every=10):
    model.eval()
    all_refs, all_hyps = [], []
    for i, batch in enumerate(dl, start=1):
        x = batch["input_features"].to(device)
        x_lens = batch["n_time_steps"].to(device)
        sess = batch["meta"][0]["session_key"]

        _, _, l3, _ = model(x, x_lens, sess)

        hyps = ctc_greedy_decode_ids(l3, x_lens, blank_id=blank_id)
        refs = get_ref_ids_from_batch(batch)

        all_hyps.extend(hyps)
        all_refs.extend(refs)

        if i % log_every == 0:
            print(f"[PER] batches={i} running={per_score(all_refs, all_hyps):.4f}")

        if max_batches is not None and i >= max_batches:
            break

    return per_score(all_refs, all_hyps)


if __name__ == "__main__":
    device = get_device()

    model = Encoder(input_dim=512, d=256, phoneme_class=41).to(device)
    criterion = Hierarchical_Loss(balance=0.3).to(device)
    for file_name in ["./_final_bal0p0_seed1_FINISHED.pt","./_final_bal0p5_seed1_FINISHED.pt"]:
        print(file_name)
        step, extra = load_checkpoint(
            model=model,
            optimizer=None,
            path=file_name,
            device=device
        )
    
        #dl_c = get_Card_Dataloader_grouped(batch_size=16, split="val", num_workers=0)
        dl_w = get_Willet_Dataloader_grouped(batch_size=16, split="train")
        try:
            blank_id = TOKENS.index("<blk>")
        except Exception:
            blank_id = 0
        print("blank_id =", blank_id)
        
        per = eval_per_no_decoder(model, dl_w, device, max_batches=20, blank_id=blank_id)
        print(f"Willet PER (no decoder): {per:.4f} ({per*100:.2f}%)")
