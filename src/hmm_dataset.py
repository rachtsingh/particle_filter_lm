import torch
from torch.utils import data
import numpy as np


def create_hmm_data(N, seq_len, x_dim, z_dim, params=None):
    from hmmlearn.hmm import MultinomialHMM  # introduces a lot of dependencies
    hmm = MultinomialHMM(n_components=z_dim)

    if params is None:
        T = np.random.random(size=(z_dim, z_dim))
        T = T/T.sum(axis=1).reshape((z_dim, 1))

        pi = np.random.random(size=(z_dim,))
        pi = pi/pi.sum()

        emit = np.random.random(size=(z_dim, x_dim))
        emit = emit/emit.sum(axis=1).reshape((z_dim, 1))
    else:
        T, pi, emit = params

    hmm.transmat_ = T
    hmm.startprob_ = pi
    hmm.emissionprob_ = emit

    X = np.zeros((N, seq_len)).astype(np.int)
    for i in range(N):
        x, _ = hmm.sample(n_samples=seq_len)
        X[i] = x.reshape((seq_len,))

    return (T, pi, emit), HMMData(X)


def sample_from_hmm(N, seq_len, hmm):
    X = np.zeros((N, seq_len)).astype(np.int)
    for i in range(N):
        x, _ = hmm.sample(n_samples=seq_len)
        X[i] = x.reshape((seq_len,))
    return HMMData(X)


class HMMData(data.Dataset):

    def __init__(self, X):
        self.data = X

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
