"""
Use this file to evaluate the quality of a learned HMM on some dataset - we can evaluate its true log marginal here.

For now, we assume that we can fit entire dataset into memory without issues.
"""

import torch
import argparse
from src.hmm import HMM

parser = argparse.ArgumentParser(description='Evaluate the log-marginal of a dataset given the parameters')
parser.add_argument('--dataset', type=str, required=True,
                    help='which dataset to evaluate')
parser.add_argument('--params', type=str, required=True,
                    help='where to load the HMMs params from')
parser.add_argument('--z-dim', type=int, default=5,
                    help='dimensionality of the hidden z')
parser.add_argument('--x-dim', type=int, default=10,
                    help='dimensionality of the observed data')
args = parser.parse_args()


def main():
    with open(args.dataset, 'r') as f:
        val_data = torch.load(f)
        if type(val_data) == tuple:
            val_data = val_data[1]
    with open(args.params, 'r') as f:
        params = torch.load(f)
    hmm = HMM(args.z_dim, args.x_dim, params=params)
    print(hmm.log_marginal(torch.from_numpy(val_data.T).contiguous()).sum())


if __name__ == '__main__':
    main()
