#!/home/rachitsingh/venv/bin/python
"""
This is to train the Mixture-of-Softmaxes models (MoS) as a discrete latent variable generative model.

Things that are missing from the reimplementation (TODO):
    - BPTT somehow
    - alpha/beta/all the ASGD stuff
    - regularization via weightdrop / embedded dropout
"""
import argparse
import time
import sys
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau  # noqa: F401
import pdb  # noqa: F401

from src.utils import get_sha, VERSION
from src.real_hmm_dataset import OneBillionWord as ChunkedDataset
from src.optim_alt import MixedOpt
from src import mos

parser = argparse.ArgumentParser(description='Demonstration of Sequential Latent VI for HMMs')
parser.add_argument('--dataset', type=str, default='1billion',
                    help='one of [generate, 1billion, <something>.pt, ...]')
parser.add_argument('--model', type=str, default='',
                    help='which generative model to use (mfvi_mos, etc.)')
parser.add_argument('--load-subset', type=str, default=None,
                    help='Which file to load inference network parameters from, if any')
parser.add_argument('--subset-type', type=str, default=None,
                    help='What subset model is loaded')
parser.add_argument('--load-model', type=str,
                    help='Just load a pretrained version of the same model, if any')

# inference model parameters
parser.add_argument('--word-dim', type=int, default=300,
                    help='dimensionality of the word embedding')
parser.add_argument('--nhid-inf', type=int, default=64,
                    help='number of hidden units per layer in the inference network')

# generative model parameters
parser.add_argument('--z-dim', type=int, default=10,
                    help='dimensionality of the hidden z')
parser.add_argument('--nhid', type=int, default=400,
                    help='number of hidden units per layer')
parser.add_argument('--nhidlast', type=int, default=300,
                    help='number of hidden units for the last rnn layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers in the generative model')
parser.add_argument('--x-dim', type=int, default=10000,
                    help='dimensionality of the observed data')

# regularization parameters
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--dropoutl', type=float, default=-0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--tied', action='store_false',
                    help='tie the word embedding and softmax weights')

# KL annealing if that's a thing you want to do - mostly turned off
parser.add_argument('--kl-anneal-delay', type=float, default=4,
                    help='number of epochs to delay increasing the KL divergence contribution')
parser.add_argument('--kl-anneal-rate', type=float, default=0.0001,
                    help='amount to increase the KL divergence amount *per batch*')
parser.add_argument('--kl-anneal-start', type=float, default=0.0001,
                    help='starting KL annealing value; upperbounds initial KL before annealing')

# optimization parameters
parser.add_argument('--epochs', type=int, default=1000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--temp', type=float, default=0.35,
                    help='temperature of the posterior to use for relaxed discrete latents')
parser.add_argument('--temp_prior', type=float, default=0.3,
                    help='temperature of the prior to use for relaxed discrete latents')
parser.add_argument('--slurm-id', type=int, help='for use in SLURM scripts')
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
parser.add_argument('--embedding', type=str, default=None,
                    help='Which file to load word embeddings from')
parser.add_argument('--load-z-gru', type=str, default=None,
                    help='Which file to use to initialize the p(z) GRU')
parser.add_argument('--base-filename', type=str, default=None)

# scheduler + optimizer
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=3.0,
                    help='gradient clipping')
parser.add_argument('--no-scheduler', action='store_true')
parser.add_argument('--optimizer', type=str, default='Adam')
args = parser.parse_args()

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
# Overload the args based on slurm id
###############################################################################
if args.slurm_id:
    print("No SLURM configuration found, write this code please")
    sys.exit(1)

if not args.quiet:
    print("running {}".format(' '.join(sys.argv)))

###############################################################################
# Load data and build the model
###############################################################################
data_kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

if args.dataset == '1billion':
    args.batch_size = 1
    train_data = ChunkedDataset('data/1_billion_word/1b-100k-train.hdf5')
    val_data = ChunkedDataset('data/1_billion_word/1b-100k-val.hdf5')
elif args.dataset == 'ptb':
    args.batch_size = 1
    train_data = ChunkedDataset('data/ptb/ptb-train.hdf5')
    val_data = ChunkedDataset('data/ptb/ptb-val.hdf5')
else:
    train_data = ChunkedDataset(args.dataset + '-train.hdf5')
    val_data = ChunkedDataset(args.dataset + '-val.hdf5')

train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                           shuffle=True, **data_kwargs)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                                         shuffle=True, **data_kwargs)


if args.model == 'mfvi_mos':
    model = mos.MFVI_Mos(args.x_dim, args.word_dim, args.nhid_inf, args.nhid, args.nhidlast, args.nlayers, args.dropout,
                         args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied, args.dropoutl, args.z_dim)
else:
    raise NotImplementedError("TODO")

if args.embedding is not None and model.load_embedding:
    data = torch.load(args.embedding)
    model.load_embedding(data)

if args.load_model:
    if not hasattr(model, 'dec') and hasattr(model, 'organize'):
        model.organize()
    model.load_state_dict(torch.load(args.load_model))

if args.load_subset:
    data = torch.load(args.load_subset)
    model.load_subset(data)

# cudafy after everything else is loaded
if args.cuda and torch.cuda.is_available():
    model.cuda()

total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())

if not args.quiet:
    print("sha: {}".format(get_sha().strip()))
    print('args:', args)
    print('model total parameters:', total_params)
    # print('model parameter breakdown:', model.get_parameter_statistics())
    print('model architecture:')
    print(model)


# Loop over epochs.
args.anneal = 0.01
lr = args.lr
best_val_loss = 1e10
true_marginal = 0
stored_loss = 100000000


def flush():
    mode = 'w' if VERSION[0] == 2 else 'wb'
    if args.save is not None:
        with open(args.save, mode) as f:
            torch.save(model.state_dict(), f)
            print('saved parameters to {}'.format(args.save))


# At any point you can hit Ctrl + C to break out of training early.
try:
    print('organized: ' + str(hasattr(model, 'dec')))
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'MixedOpt':
        optimizer = MixedOpt(model.enc.parameters(), model.dec.parameters(), args.lr, 20.0)  # this is pretty aggressive

    if not args.no_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)
    else:
        print("ignoring scheduler, lr is fixed")

    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()

        if epoch < args.kl_anneal_delay:
            args.anneal = args.kl_anneal_start

        train_loss = model.train_epoch(train_loader, optimizer, epoch, args, args.num_importance_samples)

        # let's ignore ASGD for now
        val_loss, val_nll = model.evaluate(val_loader, args, args.num_importance_samples)

        if not args.no_scheduler:
            scheduler.step(val_loss)

        print("anneal: {:.3f}".format(args.anneal))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            flush()
        if not args.quiet:
            if args.dataset not in ('1billion', 'ptb'):
                ppl = val_loss
                true_marginal_ppl = -true_marginal
            elif val_loss < 10:
                ppl = np.exp(val_loss)
                true_marginal_ppl = np.exp(-true_marginal)
            else:
                ppl = np.inf
                true_marginal_ppl = np.inf

            if not args.no_scheduler:
                scheduler.step()

            # print out the epoch summary
            print('-' * 80)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid ELBO {:5.2f} | valid NLL {:5.2f} | PPL: {:5.2f} | true PPL: {:5.2f}'
                  ''.format(epoch, (time.time() - epoch_start_time), val_loss, val_nll, ppl, true_marginal_ppl))
            print('-' * 80)
            sys.stdout.flush()
except KeyboardInterrupt:
    if not args.quiet:
        print('-' * 89)
        print('Exiting from training early')
if args.print_best:
    print("{},{}".format(-best_val_loss, true_marginal))
