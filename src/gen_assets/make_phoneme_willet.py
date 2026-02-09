from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
from g2p_en import G2p

_WORD_RE = re.compile(r"[a-zA-Z']+")
_STRESS_RE = re.compile(r"^([A-Z]+)([0-2])$")


def strip_stress(p):
    m = _STRESS_RE.match(p)
    return m.group(1) if m else p

def tokenize_words(text):
    return [w.lower() for w in _WORD_RE.findall(text)]

def load_lexicon_nostress(lexicon_path):

    lex = {}
    with open(lexicon_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            w = parts[0].lower()
            phones = [p for p in parts[1:]]
            lex[w] = phones
    return lex


@dataclass
class PhonemeTargeter:
    lexicon: Dict[str, List[str]]
    phone2id: Dict[str, int]
    sil_phone: str = "sil"
    use_sil_between_words: bool = True
    drop_oov_words: bool = True

    def __post_init__(self):
        self.g2p = G2p()
        self._g2p_cache: Dict[str, List[str]] = {}

    def _g2p_word(self, word):
        if word in self._g2p_cache:
            return self._g2p_cache[word]

        out = self.g2p(word)
        phones: List[str] = []
        for tok in out:
            if tok is None:
                continue
            t = str(tok).strip()
            if not t:
                continue

            if t == " ":
                continue

            if re.fullmatch(r"[A-Za-z]+[0-2]?", t):
                p = strip_stress(t.upper())
                phones.append(p)


        if len(phones) == 0:
            self._g2p_cache[word] = []
            return None

        self._g2p_cache[word] = phones
        return phones

    def words_to_phones(self, words):
        phones_all: List[str] = []
        for wi, w in enumerate(words):
            phones = self.lexicon.get(w)
            if phones is None:
                phones = self._g2p_word(w)
                if phones is None:
                    if self.drop_oov_words:
                        continue
                    else:
                        phones = []

            phones_all.extend(phones)

            if self.use_sil_between_words and wi != (len(words) - 1):
                phones_all.append(self.sil_phone)

        return phones_all

    def phones_to_ids(self, phones):
        ids: List[int] = []
        for p in phones:
            p_norm = p.lower() if p == self.sil_phone else p.upper()

            if p_norm in self.phone2id:
                ids.append(self.phone2id[p_norm])
            elif p.upper() in self.phone2id:
                ids.append(self.phone2id[p.upper()])
            elif p.lower() in self.phone2id:
                ids.append(self.phone2id[p.lower()])
            else:

                continue
        return ids

    def text_to_target_ids(self, text):
        words = tokenize_words(text)
        phones = self.words_to_phones(words)
        ids = self.phones_to_ids(phones)
        return ids, words, phones


def build_phone2id_from_tokens(tokens):
    return {tok: i for i, tok in enumerate(tokens)}
