import torch
from .manage_datasets.uploadDatasets import get_Card_Dataloader_grouped, get_Willet_Dataloader_grouped
from .model.decoder import *

def refs_card(batch):
    """
    Extrait les références textuelles du batch Card.
    
    batch["texts"] contient maintenant des strings décodées directement depuis le HDF5.
    """
    texts = batch.get("texts", [])
    
    # Si tous les texts sont None, le dataset n'a pas de transcriptions
    if all(t is None for t in texts):
        print(f"⚠️  WARNING: All texts are None in this batch!")
        print(f"   This usually means the split doesn't have 'transcription' field in HDF5")
        print(f"   Returning empty strings for all samples")
        return [""] * len(texts)
    
    # Convertir None en string vide
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
    dl_w_test = get_Willet_Dataloader_grouped(batch_size=16, split="val")
    wer_w, refs_w, hyps_w = eval_wer(
        model, decoder, dl_w_test, device,
        get_refs=lambda b: b["texts"],
        log_every=10,
    )
    print("Willett WER:", wer_w)
    print("REF:", refs_w[0])
    print("HYP:", hyps_w[0])

def testCard(model, decoder, device, batch_size=16):
    dl_c_test = get_Card_Dataloader_grouped(batch_size=batch_size, split="val")
    wer_c, refs_c, hyps_c = eval_wer(
        model, decoder, dl_c_test, device,
        get_refs=refs_card,
        log_every=10,
        max_batches=3,
    )
    
    # Afficher quelques exemples
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