from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from src.gen_assets.make_phoneme_willet import PhonemeTargeter

class WillettTrialDataset(Dataset):
    """
    Dataset for Willet data

    Returns:
      - x: (T_i, features)
      - text: sentence
      - meta: (file, trial index, block id)
    """

    def __init__(self, train_dir, use_area6v_only = True, cache_block_stats = True, eps = 1e-6, targeter=None, max_target_len=500):
        self.train_dir = Path(train_dir)
        self.use_area6v_only = use_area6v_only
        self.use_tx = "tx1"
        self.cache_block_stats = cache_block_stats
        self.eps = eps
        self.targeter = targeter
        self.max_target_len = max_target_len

        self.files = sorted(self.train_dir.glob("*.mat"))
        if not self.files:
            raise FileNotFoundError(f"No .mat files found in {self.train_dir}")

        self.index = [] 
        self._file_trial_counts = {}

        for fp in self.files:
            mat = sio.loadmat(fp, squeeze_me=True, struct_as_record=False)
            S = len(mat["spikePow"])
            self._file_trial_counts[fp] = S
            for i in range(S):
                self.index.append((fp, i))

        self._block_stats_cache = {}
        self._mat_cache = {}

    def __len__(self):
        return len(self.index)

    def _load_mat(self, fp):
        if fp in self._mat_cache:
            return self._mat_cache[fp]
        mat = sio.loadmat(fp, squeeze_me=True, struct_as_record=False)
        if self.cache_block_stats is False:
            pass
        return mat

    def _get_features(self, spikePow, tx, i):
        if self.use_area6v_only:
            sp = spikePow[i][:, :128] 
            t = tx[i][:, :128]       
        else:
            sp = spikePow[i]          
            t = tx[i]               
        return np.concatenate([sp, t], axis=1)  

    def _compute_block_stats_in_file(self, mat, fp):
        spikePow = mat["spikePow"]
        tx = mat[self.use_tx]
        blockIdx = mat["blockIdx"]
        S = len(spikePow)

        block_data = defaultdict(list)
        for i in range(S):
            b = int(blockIdx[i])
            block_data[b].append(self._get_features(spikePow, tx, i))

        block_stats = {}
        for b, xs in block_data.items():
            all_x = np.concatenate(xs, axis=0)
            mu = all_x.mean(axis=0)             
            std = all_x.std(axis=0) + self.eps  
            block_stats[b] = (mu, std)
        return block_stats

    def _get_block_stats(self, fp, mat):
        if fp in self._block_stats_cache:
            return self._block_stats_cache[fp]
        stats = self._compute_block_stats_in_file(mat, fp)
        if self.cache_block_stats:
            self._block_stats_cache[fp] = stats
        return stats

    def __getitem__(self, idx):
        fp, i = self.index[idx]
        mat = self._load_mat(fp)

        spikePow = mat["spikePow"]
        tx = mat[self.use_tx]
        sentenceText = mat["sentenceText"]
        blockIdx = mat["blockIdx"]

        x = self._get_features(spikePow, tx, i)

        # block-wise z-score within this file
        block_stats = self._get_block_stats(fp, mat)
        b = int(blockIdx[i])
        mu, std = block_stats[b]
        x_z = (x - mu) / std

        text = str(sentenceText[i])

        x_t = torch.from_numpy(x_z).float()

        y_t = None
        y_len = None
        if self.targeter is not None:
            y_ids, _, _ = self.targeter.text_to_target_ids(text)
            y_ids = y_ids[: self.max_target_len]
            y_len = len(y_ids)
            y_t = torch.tensor(y_ids, dtype=torch.long)

        meta = {
            "file_path": str(fp),
            "trial_idx": int(i),
            "block_idx": b,
            "session_key": fp.stem.lower().replace(".", "_"),
            "subject_key": "t12"
        }

        print(f"WILLETT Y: {y_ids}")
        print(f"WILLETT Transcription: {text}")

        return x_t, y_t, y_len, text, meta




def collate_willett_trials(batch):
    xs, ys, y_lens, trs, metas = zip(*batch)

    x_lens = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    xs_pad = pad_sequence(xs, batch_first=True, padding_value=0.0)

    # if targets exist
    if all(y is not None for y in ys):
        y_lens_t = torch.tensor([int(l) for l in y_lens], dtype=torch.long)
        ys_pad = pad_sequence(list(ys), batch_first=True, padding_value=0)
    else:
        print("Willet: Y NONE")
        ys_pad = None
        y_lens_t = None

    return {
        "input_features": xs_pad,
        "n_time_steps": x_lens,
        "seq_class_ids": ys_pad,
        "phone_seq_lens": y_lens_t,
        "texts": list(trs),
        "meta": list(metas),
    }


if __name__ == "__main__":

    phone2id = build_phone2id_from_tokens(TOKENS)
    lex = load_lexicon_nostress("data/assets/lexicon_nostress.txt")

    targeter = PhonemeTargeter(
        lexicon=lex,
        phone2id=phone2id,
        sil_phone="sil",
        use_sil_between_words=True,
        drop_oov_words=True,
    )

    ds = WillettTrialDataset("data/WilletData/train", use_area6v_only=True, targeter=targeter, max_target_len=500)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_willett_trials)

    batch = next(iter(dl))
    print(batch["input_features"].shape)
    print(batch["n_time_steps"][:5])
    print("y:", batch["seq_class_ids"].shape, "y_lens:", batch["phone_seq_lens"][:5])
    print("y min/max:", int(batch["seq_class_ids"].min()), int(batch["seq_class_ids"].max()))
    print(batch["texts"][0])
    print(batch["meta"][0])

