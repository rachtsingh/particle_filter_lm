"""
Use this file to evaluate the quality of a learned HMM on some dataset - we can evaluate its true log marginal here.

For now, we assume that we can fit entire dataset into memory without issues.
"""

import torch
import argparse
import numpy as np
from src.hmm import HMM
from src import hmm, hmm_filter
from src.real_hmm_dataset import OneBillionWord

parser = argparse.ArgumentParser(description='Evaluate the log-marginal of a dataset given the parameters')
parser.add_argument('--dataset', type=str, required=True,
                    help='which dataset to evaluate')
parser.add_argument('--params', type=str, default=None,
                    help='where to load the HMMs params from')
parser.add_argument('--model', type=str, default='hmm_em',
                    help='which model to load: [hmm_vi, hmm_em, ...]')

# model configuration parameters
parser.add_argument('--nhid', type=int, default=32,
                    help='number of hidden units per layer')
parser.add_argument('--z-dim', type=int, default=5,
                    help='dimensionality of the hidden z')
parser.add_argument('--x-dim', type=int, default=10,
                    help='dimensionality of the observed data')
parser.add_argument('--hidden', type=int, default=10,
                    help='dimensionality of hidden size in generative model, if any')
parser.add_argument('--temp', type=float, default=0.8,
                    help='temperature of the posterior to use for relaxed discrete latents')
parser.add_argument('--temp_prior', type=float, default=0.5,
                    help='temperature of the prior to use for relaxed discrete latents')
parser.add_argument('--num-importance-samples', type=int, default=5,
                    help='number of samples to take for IWAE')
parser.add_argument('--no-cuda', action='store_true',
                    help='use CUDA')

# whether to load from a trajectory
parser.add_argument('--traj', type=str, default=None,
                    help='trajectory file to load if any, will save to file with same name but .npy extension')
args = parser.parse_args()

if torch.cuda.is_available():
    args.cuda = not args.no_cuda
else:
    args.cuda = False

if args.params is None and args.traj is None:
    raise ValueError("must specify either --params or --traj")


def eval_hmm(model, data):
    return model.log_marginal(torch.from_numpy(data.T).contiguous()).sum()


def eval_model(model, data, iterate_val=False):
    if not iterate_val:
        return model.eval_log_marginal(torch.from_numpy(data.T).contiguous()).sum()
    else:
        val_loss, true_marginal = model.evaluate(data, args, args.num_importance_samples)
        return true_marginal


def main():
    if args.dataset == '1billion':
        val_data = OneBillionWord('data/1_billion_word/1b-100k-val.hdf5')
        data_kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
        val_data = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, **data_kwargs)
        iterate_val = True
    else:
        with open(args.dataset, 'r') as f:
            val_data = torch.load(f)
            if type(val_data) == tuple:
                val_data = val_data[1]
        iterate_val = False

    if args.params:
        if args.model == 'hmm_vi':
            model = hmm_filter.HMM_VI(z_dim=args.z_dim, x_dim=args.x_dim, nhid=args.nhid,
                                      temp=args.temp, temp_prior=args.temp_prior, params=None)
        elif args.model == 'hmm_em':
            model = hmm.HMM_EM(args.z_dim, args.x_dim)
        elif args.model == 'hmm_deep_em':
            model = hmm.HMM_EM_Layers(args.z_dim, args.x_dim, args.hidden)
        if args.cuda:
            model = model.cuda()
        print(eval_model(model, val_data, iterate_val))

    elif args.traj:
        print(eval)
        data = np.load(args.traj)
        T, pi, emit = data['T'], data['pi'], data['emit']
        epochs = T.shape[0]
        vals = np.zeros(epochs)
        for i in range(epochs):
            hmm_model = HMM(args.z_dim, args.x_dim, params=(T[i], pi[i], emit[i]))
            vals[i] = eval_hmm(hmm_model, val_data)
            print(vals[i])
        np.save('{}_eval'.format(args.traj.split('.')[0]), vals)


if __name__ == '__main__':
    main()
