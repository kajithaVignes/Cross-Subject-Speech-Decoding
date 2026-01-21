import glob
import os
import scipy.io as sio
import numpy as np
from collections import defaultdict

TRAIN_DIR = "data/WilletData/train"
test_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.mat")))

def load_one_mat(path):
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    spikePow = mat["spikePow"]
    tx1 = mat["tx1"]
    sentenceText = mat["sentenceText"]
    blockIdx = mat["blockIdx"]
    return spikePow, tx1, sentenceText, blockIdx

def get_features(spikePow, tx1, i):
    """
    Extract features from each trial
    
    :param spikePow: Gamma Band Power
    :param tx1: Threshold crossing
    :param i: trial index
    """
    sp = spikePow[i][:, :128]  # (T_i, 128) As suggested in the README file of the dataset, we only use the first 128 features, corresponding to 6v neural activity
    tx = tx1[i][:, :128]       # (T_i, 128)
    return np.concatenate([sp, tx], axis=1)  # (T_i, 256)

def compute_block_stats_in_file(spikePow, tx1, blockIdx, eps=1e-6):
    """
    For each block, compute mean and std of features across all trials

    """
    S = len(spikePow)
    block_data = defaultdict(list)
    for i in range(S):
        b = int(blockIdx[i])
        block_data[b].append(get_features(spikePow, tx1, i)) # Groups trials by block

    block_stats = {}
    for b, xs in block_data.items(): # Compute mean and std for each block
        all_x = np.concatenate(xs, axis=0)   # (sum_T, 256)
        mu = all_x.mean(axis=0)              # (256,)
        std = all_x.std(axis=0) + eps        # (256,)
        block_stats[b] = (mu, std)
    return block_stats

def iter_train_trials():
    for path in test_files:
        spikePow, tx1, sentenceText, blockIdx = load_one_mat(path)
        S = len(spikePow)

        block_stats = compute_block_stats_in_file(spikePow, tx1, blockIdx)

        for i in range(S): # yield normalized trials
            x = get_features(spikePow, tx1, i)
            mu, std = block_stats[int(blockIdx[i])]
            x_z = (x - mu) / std
            y_text = sentenceText[i]
            yield x_z, y_text, int(blockIdx[i]), path

if __name__ == "__main__":
    x_z, y_text, b, path = next(iter_train_trials())
    print("From:", path)
    print("Block:", b)
    print("x_z shape:", x_z.shape)
    print("sentenceText entry :", y_text)

