import argparse
import time
import sys
import math
import numpy as np
import torch
import pdb  # noqa: F401

from src.utils import get_sha
from src.hmm_dataset import create_hmm_data
from src import hmm_filter

parser = argparse.ArgumentParser(description='Demonstration of Sequential Latent VI for HMMs')
parser.add_argument('--dataset', type=str, default='generate',
                    help='one of [generate, ...]')
parser.add_argument('--nhid', type=int, default=1024,
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
parser.add_argument('--temp', type=float, default=0.6,
                    help='temperature of the posterior to use for relaxed discrete latents')
# parser.add_argument('--temp_prior', type=float, default=0.4,
#                     help='temperature of the prior to use for relaxed discrete latents')
parser.add_argument('--num-importance-samples', type=int, default=5,
                    help='number of samples to take for IWAE')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no-cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--prof', type=str, default=None,
                    help='If specified, profile the first 10 batches and dump to <prof>')
args = parser.parse_args()

print("running {}".format(' '.join(sys.argv)))

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    args.cuda = not args.no_cuda
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
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

if args.dataset == 'generate':
    params, train_data = create_hmm_data(N=100, seq_len=20, x_dim=args.x_dim, z_dim=args.z_dim)
    val_data = create_hmm_data(N=100, seq_len=15, x_dim=args.x_dim, z_dim=args.z_dim, params=params)
else:
    raise NotImplementedError("that dataset is not implemented")

# build the model using the true parameters of the generative model
model = hmm_filter.HMMInference(z_dim=args.z_dim, x_dim=args.x_dim, nhid=args.nhid, params=params)

if args.cuda and torch.cuda.is_available():
    model.cuda()

total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())

print("sha: {}".format(get_sha().strip()))
print('args:', args)
print('model total parameters:', total_params)
print('model architecture:')
print(model)

# Loop over epochs.
args.anneal = 0.01
lr = args.lr
best_val_loss = []
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()

        if epoch < args.kl_anneal_delay:
            args.anneal = 0.0001

        train_loss = model.train_epoch(train_data, optimizer, epoch, args, args.num_importance_samples)

        # let's ignore ASGD for now
        val_loss, val_elbo, val_nll = model.evaluate(val_data, args, True, args.num_importance_samples)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid IWAE {:5.2f} | valid ELBO {:5.2f} | valid NLL {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), val_loss, val_elbo, val_nll,
                                         math.exp(val_loss) if val_loss < 10. else float('inf')))
        print('-' * 89)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
