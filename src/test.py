"""
This file tests my implementation of the forward-backward using the (old) sklearn implementation.
My version is also vectorized, which is kind of nice, but it isn't careful about log values so there's
a lot of numerical imprecision.
"""

from __future__ import print_function
import datetime
import torch
import numpy as np
from matplotlib import cm, pyplot as plt
from hmmlearn.hmm import MultinomialHMM
from pytorch_hmm import HMM
import pdb

def main():
    hmm = MultinomialHMM(n_components=5)
    T = np.random.random(size=(5, 5))
    T = T/T.sum(axis=1).reshape((5, 1))
    hmm.transmat_ = T
    
    pi = np.random.random(size=(5,))
    pi = pi/pi.sum()
    hmm.startprob_ = pi

    emit = np.random.random(size=(5, 10))
    emit = emit/emit.sum(axis=1).reshape((5, 1))
    hmm.emissionprob_ = emit

    X = np.zeros((20, 25)).astype(np.int)
    for i in range(20):
        x, _ = hmm.sample(n_samples=25)
        X[i] = x.reshape((25,))

    # load the PyTorch HMM
    phmm = HMM(z_dim=5, x_dim=10)
    phmm.T = torch.Tensor(T.T)
    phmm.pi = torch.Tensor(pi)
    phmm.emit = torch.Tensor(emit.T)

    # compute PyTorch HMM forward-backward 
    my_marginals = phmm.log_marginal(torch.Tensor(X.T))

    # compute hmmlearn version
    true_marginals = np.zeros(20)
    for i in range(20):
        true_marginals[i] = hmm.score(X[i].reshape((-1, 1)))

    assert np.abs(true_marginals - my_marginals.numpy()).max() < 1e-4

if __name__ == '__main__':
    main()