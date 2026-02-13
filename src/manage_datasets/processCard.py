from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

SPLITS = ("train", "val", "test")

def _split_from_filename(name):
    n = name.lower()
    if "data_train" in n:
        return "train"
    if "data_val" in n:
        return "val"
    if "data_test" in n:
        return "test"
    return "unknown"

class CardT15TrialDataset(Dataset):
    """
    Dataset for Card

    Returns:
      - x: (T_i, F) : neural features (often F=512)
      - y: (L,) or None : seq_class_ids (phoneme labels) if present
      - t: str or None : transcription text if present
      - meta: file/session/trial metadata
    """

    def __init__(self, data_root, split = "train", feature_subset = None, verify_keys = True):
        
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split}")

        self.data_root = Path(data_root)
        self.split = split
        self.feature_subset = feature_subset

        # Collect files matching the split
        candidates = list(self.data_root.rglob(f"data_{split}.hdf5")) + list(self.data_root.rglob(f"data_{split}.hd5f"))
        self.files = sorted(set(candidates))

        if len(self.files) == 0:
            raise FileNotFoundError(f"No data_{split}.hdf5/.hd5f found under {self.data_root}")

        # Build an index of (file_path, trial_key)
        self.index: List[Tuple[Path, str]] = []
        for fp in self.files:
            with h5py.File(fp, "r") as f:
                trial_keys = sorted([k for k in f.keys() if k.startswith("trial_")])
                self.index.extend([(fp, k) for k in trial_keys])

                if verify_keys and len(trial_keys) > 0:
                    g = f[trial_keys[0]]
                    if "input_features" not in g:
                        raise KeyError(f"{fp} {trial_keys[0]} missing 'input_features' dataset")

        if len(self.index) == 0:
            raise RuntimeError(f"Found split files, but no trial_{split} groups under them: {self.files[:3]} ...")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        fp, trial_key = self.index[idx]
        session_dir = fp.parent

        with h5py.File(fp, "r") as f:
            g = f[trial_key]

            x = g["input_features"][:]  # (T_i, F)
            if self.feature_subset is not None:
                x = x[:, self.feature_subset]

            # Optional labels (not in test)
            y = g["seq_class_ids"][:] if "seq_class_ids" in g else None
            
            # Transcription: decode ASCII bytes to text string
            tr_text = None
            if "transcription" in g:
                tr_array = g["transcription"][:]
                # Remove padding and decode ASCII bytes
                non_zero = tr_array[tr_array != 0]
                if len(non_zero) > 0:
                    try:
                        tr_text = bytes(non_zero).decode('utf-8', errors='ignore').strip()
                    except Exception as e:
                        print(f"Warning: failed to decode transcription in {trial_key}: {e}")
                        tr_text = None
            
            if tr_text is None and "sentence_label" in g.attrs:
                tr_text = str(g.attrs["sentence_label"])

            n_time_steps = int(g.attrs.get("n_time_steps", x.shape[0]))
            seq_len = int(g.attrs.get("seq_len", 0))
            block_num = int(g.attrs.get("block_num", -1))
            trial_num = int(g.attrs.get("trial_num", -1))

        x_t = torch.from_numpy(x).float()
        y_t = torch.from_numpy(y).long() if y is not None else None

        print(f"CARD Y: {y}")
        print(f"CARD Transcription: {tr_text}")

        meta = {
            "split": self.split,
            "file_path": str(fp),
            "session_dir": str(session_dir),
            "trial_key": trial_key,
            "n_time_steps": n_time_steps,
            "seq_len": seq_len,
            "block_num": block_num,
            "trial_num": trial_num,
            "session_key": Path(fp).parent.name.lower().replace(".", "_"),
            "subject_key": "t15"
        }

        return x_t, y_t, tr_text, meta

def infer_day_key_from_card_meta(meta):
    sess = Path(meta["session_dir"]).name.lower()
    return f"day_{sess}"

def collate_card_trials(batch):
    """
    batch: list of (x, y, tr_text, meta)
    Returns padded tensors + lengths.
    """
    xs, ys, tr_texts, metas = zip(*batch)

    x_lens = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    xs_pad = pad_sequence(xs, batch_first=True, padding_value=0.0)  # (B, T_max, F)

    if all(y is not None for y in ys):
        y_lens = torch.tensor([y.shape[0] for y in ys], dtype=torch.long)
        ys_pad = pad_sequence(list(ys), batch_first=True, padding_value=0)  # (B, L_max)
    else:
        y_lens = None
        ys_pad = None

    return {
        "input_features": xs_pad,
        "n_time_steps": x_lens,
        "seq_class_ids": ys_pad,
        "phone_seq_lens": y_lens,
        "texts": list(tr_texts),
        "meta": list(metas),
    }


if __name__ == "__main__":

    ds_train = CardT15TrialDataset(data_root="data/CardData/hdf5_data_final", split="train", feature_subset=None)
    dl_train = DataLoader(ds_train, batch_size=16, shuffle=True, num_workers=0, collate_fn=collate_card_trials)

    print(len(ds_train))

    batch = next(iter(dl_train))
    print(batch["input_features"].shape)
    print(batch["n_time_steps"][:5])
    print("Texts:", batch["texts"][:3])