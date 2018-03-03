"""
Use this file to evaluate the quality of a learned HMM on some dataset - we can evaluate its true log marginal here.

For now, we assume that we can fit entire dataset into memory without issues.
"""

import torch
import argparse
import numpy as np
from src.hmm import HMM

parser = argparse.ArgumentParser(description='Evaluate the log-marginal of a dataset given the parameters')
parser.add_argument('--dataset', type=str, required=True,
                    help='which dataset to evaluate')
parser.add_argument('--params', type=str, default=None,
                    help='where to load the HMMs params from')
parser.add_argument('--z-dim', type=int, default=5,
                    help='dimensionality of the hidden z')
parser.add_argument('--x-dim', type=int, default=10,
                    help='dimensionality of the observed data')
parser.add_argument('--traj', type=str, default=None,
                    help='trajectory file to load if any, will save to file with same name but .npy extension')
args = parser.parse_args()

if args.params is None and args.traj is None:
    raise ValueError("must specify either --params or --traj")


def eval_hmm(hmm, data):
    return hmm.log_marginal(torch.from_numpy(data.T).contiguous()).sum()


def main():
    with open(args.dataset, 'r') as f:
        val_data = torch.load(f)
        if type(val_data) == tuple:
            val_data = val_data[1]

    if args.params:
        with open(args.params, 'r') as f:
            params = torch.load(f)
        hmm = HMM(args.z_dim, args.x_dim, params=params)
        print(eval_hmm(hmm, val_data))

    elif args.traj:
        print(eval)
        data = np.load(args.traj)
        T, pi, emit = data['T'], data['pi'], data['emit']
        epochs = T.shape[0]
        vals = np.zeros(epochs)
        for i in range(epochs):
            hmm = HMM(args.z_dim, args.x_dim, params=(T[i], pi[i], emit[i]))
            vals[i] = eval_hmm(hmm, val_data)
            print(vals[i])
        np.save('{}_eval'.format(args.traj.split('.')[0]), vals)


if __name__ == '__main__':
    main()
