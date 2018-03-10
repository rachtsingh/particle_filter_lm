import torch
from src.hmm_dataset import create_hmm_data
import argparse

parser = argparse.ArgumentParser(description='Generate data from HMMs')
parser.add_argument('--save-data', type=str, required=True,
                    help='a path to save the data to')
parser.add_argument('--save-params', type=str, default=None,
                    help='a path to save params to (if any)')
parser.add_argument('-N', type=int, default=1000,
                    help='size of the training dataset')
parser.add_argument('-N_val', type=int, default=100,
                    help='size of the validation dataset')
parser.add_argument('--z-dim', type=int, default=5,
                    help='dimensionality of the hidden z')
parser.add_argument('--x-dim', type=int, default=10,
                    help='dimensionality of the observed data')
args = parser.parse_args()


def generate_hmm_data(N, N_val, x_dim, z_dim):
    params, train_data = create_hmm_data(N, seq_len=10, x_dim=x_dim, z_dim=z_dim, params=None)
    _, val_data = create_hmm_data(N_val, seq_len=5, x_dim=x_dim, z_dim=z_dim, params=params)
    return train_data, val_data, params


def main():
    train_data, val_data, params = generate_hmm_data(args.N, args.N_val, args.x_dim, args.z_dim)
    with open(args.save_data, 'w') as f:
        torch.save((train_data.data, val_data.data), f)
    if not (args.save_params is None):
        with open(args.save_params, 'w') as f:
            torch.save(params, f)


if __name__ == '__main__':
    main()
