"""
Comparison of EM and VI
"""
import argparse
import time
import sys
import numpy as np
import torch
from torch import nn
import pdb  # noqa: F401

from src.utils import get_sha
from src.hmm_dataset import create_hmm_data, HMMData
from src import hmm_filter
from src import hmm

parser = argparse.ArgumentParser(description='Demonstration of Sequential Latent VI for HMMs')
parser.add_argument('--dataset', type=str, default='generate',
                    help='one of [generate, ...]')
parser.add_argument('--inference', type=str, default='vi',
                    help='which inference method to use (vi, em)')
parser.add_argument('--load-hmm', type=str,
                    help='which PyTorch file to load the HMM from, if any')
parser.add_argument('--nhid', type=int, default=32,
                    help='number of hidden units per layer')
parser.add_argument('--z-dim', type=int, default=5,
                    help='dimensionality of the hidden z')
parser.add_argument('--x-dim', type=int, default=10,
                    help='dimensionality of the observed data')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=3.0,
                    help='gradient clipping')
parser.add_argument('--kl-anneal-delay', type=float, default=4,
                    help='number of epochs to delay increasing the KL divergence contribution')
parser.add_argument('--kl-anneal-rate', type=float, default=0.0001,
                    help='amount to increase the KL divergence amount *per batch*')
parser.add_argument('--kl-anneal-start', type=float, default=0.0001,
                    help='starting KL annealing value; upperbounds initial KL before annealing')
parser.add_argument('--epochs', type=int, default=1000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--temp', type=float, default=0.8,
                    help='temperature of the posterior to use for relaxed discrete latents')
parser.add_argument('--temp_prior', type=float, default=0.5,
                    help='temperature of the prior to use for relaxed discrete latents')
parser.add_argument('--num-importance-samples', type=int, default=5,
                    help='number of samples to take for IWAE')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no-cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default=None,
                    help='path to save the final model')
parser.add_argument('--prof', type=str, default=None,
                    help='If specified, profile the first 10 batches and dump to <prof>')
parser.add_argument('--filter', action='store_true',
                    help='Turn on particle filtering')
parser.add_argument('--quiet', action='store_true',
                    help='Turn off printing except where enabled by another CLI flag')
parser.add_argument('--print-best', action='store_true',
                    help='Print the best validation loss, along with log marginal')
parser.add_argument('--dump-param-traj', type=str, default=None,
                    help='A place to dump out the parameter trajectories as an npz')
args = parser.parse_args()

if not args.quiet:
    print("running {}".format(' '.join(sys.argv)))

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    args.cuda = not args.no_cuda
    if not args.cuda:
        if not args.quiet:
            print("WARNING: You have a CUDA device, so you should probably run without --no-cuda")
        device = -1
    else:
        torch.cuda.manual_seed(args.seed)
        device = torch.cuda.current_device()
else:
    args.cuda = False
    device = -1

###############################################################################
# Load data and build the model
###############################################################################
data_kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

if args.dataset == 'generate':
    if args.load_hmm:
        params = torch.load(args.load_hmm)
    else:
        params = None
    params, train_data = create_hmm_data(N=1000, seq_len=20, x_dim=args.x_dim, z_dim=args.z_dim, params=params)
    _, val_data = create_hmm_data(N=100, seq_len=15, x_dim=args.x_dim, z_dim=args.z_dim, params=params)
    # maybe use params here to get the 'true' HMM
else:
    # we'll load the dataset from the specified file
    params = None
    train, val = torch.load(args.dataset)
    train_data = HMMData(train)
    val_data = HMMData(val)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                           shuffle=True, **data_kwargs)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                                         shuffle=True, **data_kwargs)


params = None
# build the model using the true parameters of the generative model
if args.inference == 'vi':
    model = hmm_filter.HMM_VI(z_dim=args.z_dim, x_dim=args.x_dim, nhid=args.nhid,
                              temp=args.temp, temp_prior=args.temp_prior, params=None)
elif args.inference == 'em':
    model = hmm.HMM_EM(args.z_dim, args.x_dim)

if args.cuda and torch.cuda.is_available():
    model.cuda()

total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())

if not args.quiet:
    print("sha: {}".format(get_sha().strip()))
    print('args:', args)
    print('model total parameters:', total_params)
    print('model architecture:')
    print(model)

if args.dump_param_traj:
    T_traj = np.zeros((args.epochs, args.z_dim, args.z_dim))
    pi_traj = np.zeros((args.epochs, args.z_dim))
    emit_traj = np.zeros((args.epochs, args.z_dim, args.x_dim))

# Loop over epochs.
args.anneal = 0.01
lr = args.lr
best_val_loss = 1e10
true_marginal = 0
stored_loss = 100000000


def flush():
    if args.save is not None:
        if args.inference in ('vi', 'em'):
            with open(args.save, 'w') as f:
                T = nn.Softmax(dim=0)(model.T).data.numpy().T
                pi = nn.Softmax(dim=0)(model.pi).data.numpy()
                emit = nn.Softmax(dim=0)(model.emit).data.numpy().T
                params = (T, pi, emit)
                torch.save(params, f)
                print('saved parameters to {}'.format(args.save))
    if args.dump_param_traj is not None:
        np.savez(args.dump_param_traj, T=T_traj, pi=pi_traj, emit=emit_traj)


# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()

        if epoch < args.kl_anneal_delay:
            args.anneal = args.kl_anneal_start

        train_loss = model.train_epoch(train_loader, optimizer, epoch, args, args.num_importance_samples)

        # let's ignore ASGD for now
        val_loss, true_marginal = model.evaluate(val_loader, args, args.num_importance_samples)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if not args.quiet:
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid ELBO {:5.2f} | true marginal {:5.2f}'
                  ''.format(epoch, (time.time() - epoch_start_time), val_loss, true_marginal))
            print('-' * 89)
        if args.dump_param_traj:
            T = nn.Softmax(dim=0)(model.T).data.numpy().T
            pi = nn.Softmax(dim=0)(model.pi).data.numpy()
            emit = nn.Softmax(dim=0)(model.emit).data.numpy().T
            T_traj[epoch - 1] = T
            pi_traj[epoch - 1] = pi
            emit_traj[epoch - 1] = emit

        if epoch % 10 == 0:
            args.lr = args.lr * 0.8
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
    flush()
except KeyboardInterrupt:
    if not args.quiet:
        print('-' * 89)
        print('Exiting from training early')
    flush()
if args.print_best:
    print("{},{}".format(-best_val_loss, true_marginal))
