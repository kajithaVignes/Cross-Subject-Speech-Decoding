from collections import defaultdict
from torch.utils.data import Sampler
import random
import math
from torch.utils.data import DataLoader

from .processCard import CardT15TrialDataset, collate_card_trials
from .processWillet import WillettTrialDataset, collate_willett_trials

class IndexGroupedBatchSampler(Sampler):
    """
    Groups by key computed from dataset.index[idx] (no dataset[idx] calls).
    """
    def __init__(self, dataset, batch_size: int, key_from_index, shuffle=True, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        groups = defaultdict(list)
        for idx in range(len(dataset)):
            k = key_from_index(dataset, idx)
            groups[k].append(idx)

        self.groups = dict(groups)
        self.group_keys = list(self.groups.keys())

    def __iter__(self):
        keys = self.group_keys[:]
        if self.shuffle:
            random.shuffle(keys)

        for k in keys:
            inds = self.groups[k][:]
            if self.shuffle:
                random.shuffle(inds)

            n = len(inds)
            end = n - (n % self.batch_size) if self.drop_last else n
            for i in range(0, end, self.batch_size):
                batch = inds[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self):
        total = 0
        for inds in self.groups.values():
            if self.drop_last:
                total += len(inds) // self.batch_size
            else:
                total += math.ceil(len(inds) / self.batch_size)
        return total



def get_Card_Dataloader_grouped(batch_size=16, split="train"):
    ds = CardT15TrialDataset(data_root="data/CardData/hdf5_data_final", split=split)

    def key_from_index(dataset, idx):
        fp, trial_key = dataset.index[idx]   # fp is a Path to data_train.hdf5
        return fp.parent.name.lower()        # t15.YYYY.MM.DD

    sampler = IndexGroupedBatchSampler(ds, batch_size=batch_size, key_from_index=key_from_index)
    return DataLoader(ds, batch_sampler=sampler, num_workers=0, collate_fn=collate_card_trials)


def get_Willet_Dataloader_grouped(batch_size=8, split="train"):
    ds = WillettTrialDataset(f"data/WilletData/{split}", use_area6v_only=True)

    def key_from_index(dataset, idx):
        fp, i = dataset.index[idx]     # fp is a Path to t12.YYYY.MM.DD.mat
        return fp.stem.lower()         # t12.YYYY.MM.DD

    sampler = IndexGroupedBatchSampler(ds, batch_size=batch_size, key_from_index=key_from_index)
    return DataLoader(ds, batch_sampler=sampler, num_workers=0, collate_fn=collate_willett_trials)


def mixed_batch_iterator(card_loader, willett_loader):
    it_c = iter(card_loader)
    it_w = iter(willett_loader)

    while True:
        # Card
        try:
            yield "card", next(it_c)
        except StopIteration:
            it_c = iter(card_loader)
            yield "card", next(it_c)

        # Willett
        try:
            yield "willett", next(it_w)
        except StopIteration:
            it_w = iter(willett_loader)
            yield "willett", next(it_w)



class MixedBatchIterator:
    """
    Yield batches from multiple dataloaders.
    Strategy:
      - "alternate": C, W, C, W...
      - "random": choose source each step with probs
    """
    def __init__(self, loaders: dict, strategy="alternate", probs=None, seed=0):
        self.loaders = loaders
        self.strategy = strategy
        self.probs = probs
        self.rng = random.Random(seed)

    def __iter__(self):
        iters = {k: iter(v) for k, v in self.loaders.items()}
        keys = list(self.loaders.keys())

        step = 0
        while True:
            if self.strategy == "alternate":
                key = keys[step % len(keys)]
            else:  # random
                assert self.probs is not None
                key = self.rng.choices(keys, weights=[self.probs[k] for k in keys], k=1)[0]

            try:
                batch = next(iters[key])
            except StopIteration:
                iters[key] = iter(self.loaders[key])
                batch = next(iters[key])

            batch["_source"] = key
            yield batch
            step += 1


