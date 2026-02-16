from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from jiwer import wer
import re 
import torch
import torch.nn.functional as F
import torchaudio.models.decoder

TOKENS = [
  "<blk>",
  "AA","AE","AH","AO","AW","AY","B","CH","D","DH","EH","ER","EY","F","G",
  "HH","IH","IY","JH","K","L","M","N","NG","OW","OY","P","R","S","SH",
  "T","TH","UH","UW","V","W","Y","Z","ZH",
  "sil",  
]


def build_ta_ctc_decoder(lexicon_path, arpa_path, tokens, beam_size=50, beam_threshold=20.0, lm_weight=2.0, word_score=1.0, blank_token="<blk>"):
    return torchaudio.models.decoder.ctc_decoder(
        lexicon=lexicon_path,
        tokens=tokens,
        lm=arpa_path,
        nbest=1,
        beam_size=beam_size,
        beam_threshold=beam_threshold,
        lm_weight=lm_weight,
        word_score=word_score,
        blank_token=blank_token,
        sil_token="sil"
    )


@torch.no_grad()
def ta_decode_batch(decoder, logits, lengths):
    logp = F.log_softmax(logits.float(), dim=-1).to(torch.float32).cpu()
    lengths = lengths.to(torch.int64).cpu()
    hyps = decoder(logp, lengths)
    return [" ".join(hyps[b][0].words) for b in range(len(hyps))]

@torch.no_grad()
def decode_words(decoder, logits, lengths):
    logp = torch.nn.functional.log_softmax(logits.float(), dim=-1).cpu()
    lengths = lengths.to(torch.int64).cpu()
    hyps = decoder(logp, lengths)
    return [" ".join(hyps[b][0].words) for b in range(len(hyps))]

def norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9'\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def compute_wer(refs, hyps):
    refs = [norm(r) for r in refs]
    hyps = [norm(h) for h in hyps]
    return wer(refs, hyps)
