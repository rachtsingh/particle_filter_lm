"""
These models are not HMMs, so there's no way to do EM here, only VI

The idea is that this is for bootstrapping inference on VI-based models
"""
import argparse
import time
import sys
import numpy as np
import torch
from itertools import product
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau  # noqa: F401
import pdb  # noqa: F401

from src.utils import get_sha
from src.hmm_dataset import create_hmm_data, HMMData
from src.real_hmm_dataset import OneBillionWord
from src import vi_filter

parser = argparse.ArgumentParser(description='Demonstration of Sequential Latent VI for HMMs')
parser.add_argument('--dataset', type=str, default='1billion',
                    help='one of [generate, 1billion, <something>.pt, ...]')
parser.add_argument('--model', type=str, default='',
                    help='which generative model to use (hmm_gru_mfvi, etc.)')
parser.add_argument('--load-inference', type=str, default=None,
                    help='Which file to load inference network parameters from, if any')
parser.add_argument('--load-hmm', type=str,
                    help='which PyTorch file to load the HMM from, if any')

# model config
parser.add_argument('--word-dim', type=int, default=300,
                    help='dimensionality of the word embedding for 1billion')
parser.add_argument('--nhid', type=int, default=32,
                    help='number of hidden units per layer in the inference network')
parser.add_argument('--z-dim', type=int, default=5,
                    help='dimensionality of the hidden z')
parser.add_argument('--x-dim', type=int, default=10,
                    help='dimensionality of the observed data')
parser.add_argument('--hidden', type=int, default=10,
                    help='dimensionality of hidden size in generative model, if any')
parser.add_argument('--lstm-sz', type=int, default=200,
                    help='dimensionality of the LSTM in the LSTM + HMM models')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=3.0,
                    help='gradient clipping')

# KL annealing if that's a thing you want to do
parser.add_argument('--kl-anneal-delay', type=float, default=4,
                    help='number of epochs to delay increasing the KL divergence contribution')
parser.add_argument('--kl-anneal-rate', type=float, default=0.0001,
                    help='amount to increase the KL divergence amount *per batch*')
parser.add_argument('--kl-anneal-start', type=float, default=0.0001,
                    help='starting KL annealing value; upperbounds initial KL before annealing')

#
parser.add_argument('--epochs', type=int, default=1000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--temp', type=float, default=0.3,
                    help='temperature of the posterior to use for relaxed discrete latents')
parser.add_argument('--temp_prior', type=float, default=0.35,
                    help='temperature of the prior to use for relaxed discrete latents')
parser.add_argument('--slurm-id', type=int, help='for use in SLURM scripts')
parser.add_argument('--num-importance-samples', type=int, default=5,
                    help='number of samples to take for IWAE')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no-cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default=None,
                    help='path to save the final model')
parser.add_argument('--save-params', type=str, default=None,
                    help='path to save HMM parameters')
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
    temps = [0.7, 0.8, 0.9]
    temp_priors = [0.05, 0.4, 0.75]
    settings = list(filter(lambda x: x[0] < x[1], product(temp_priors, temps)))
    if args.slurm_id >= len(settings):
        raise ValueError("must have ID < {}".format(len(settings)))
    args.temp_prior, args.temp = settings[args.slurm_id]


if not args.quiet:
    print("running {}".format(' '.join(sys.argv)))

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
elif args.dataset == '1billion':
    params = None
    args.batch_size = 1
    train_data = OneBillionWord('data/1_billion_word/1b-100k-train.hdf5')
    val_data = OneBillionWord('data/1_billion_word/1b-100k-val.hdf5')
else:
    # we'll load the dataset from the specified file
    params = None
    train, val = torch.load(args.dataset)
    train_data = HMMData(train)
    val_data = HMMData(val)

# TODO: just changed shuffle value for testing!
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                           shuffle=True, **data_kwargs)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                                         shuffle=True, **data_kwargs)


params = None

# build the model using the true parameters of the generative model
if args.model == 'hmm_gru_mfvi':
    model = vi_filter.HMM_GRU_MFVI(z_dim=args.z_dim, x_dim=args.x_dim, hidden_size=args.hidden, nhid=args.nhid,
                                   word_dim=args.word_dim, temp=args.temp, temp_prior=args.temp_prior, params=None)
elif args.model == 'hmm_gru_mfvi_deep':
    model = vi_filter.HMM_GRU_MFVI_Deep(z_dim=args.z_dim, x_dim=args.x_dim, hidden_size=args.hidden, nhid=args.nhid,
                                        word_dim=args.word_dim, temp=args.temp, temp_prior=args.temp_prior, params=None)
elif args.model == 'hmm_gru_auto_deep':
    model = vi_filter.HMM_GRU_Auto_Deep(z_dim=args.z_dim, x_dim=args.x_dim, hidden_size=args.hidden, nhid=args.nhid,
                                        word_dim=args.word_dim, temp=args.temp, temp_prior=args.temp_prior, params=None)
elif args.model == 'hmm_lstm_auto_deep':
    model = vi_filter.HMM_LSTM_Auto_Deep(z_dim=args.z_dim, x_dim=args.x_dim, hidden_size=args.hidden, nhid=args.nhid,
                                         word_dim=args.word_dim, temp=args.temp, temp_prior=args.temp_prior, params=None)
elif args.model == 'vrnn_lstm_concrete':
    model = vi_filter.VRNN_LSTM_Auto_Concrete(z_dim=args.z_dim, x_dim=args.x_dim, hidden_size=args.hidden, nhid=args.nhid,
                                              word_dim=args.word_dim, temp=args.temp, temp_prior=args.temp_prior, params=None)
elif args.model == 'hmm_lstm_sep':
    model = vi_filter.HMM_Joint_LSTM(z_dim=args.z_dim, x_dim=args.x_dim, hidden_size=args.hidden, lstm_hidden_size=args.lstm_sz,
                                     word_dim=args.word_dim, separate_opt=True)
elif args.model == 'hmm_lstm_joint':
    model = vi_filter.HMM_Joint_LSTM(z_dim=args.z_dim, x_dim=args.x_dim, hidden_size=args.hidden, lstm_hidden_size=args.lstm_sz,
                                     word_dim=args.word_dim, separate_opt=False)
else:
    raise NotImplementedError("TODO")

if args.embedding is not None and model.load_embedding:
    data = torch.load(args.embedding)
    model.load_embedding(data)

# load parameters if we want that comparison:
if args.load_hmm:
    print("loading HMM parameters from {}".format(args.load_hmm))
    model.set_params(torch.load(args.load_hmm))

if args.load_inference:
    model.init_inference(torch.load(args.load_inference))

if args.load_z_gru:
    model.init_z_gru(torch.load(args.load_z_gru))

# cudafy after everything else is loaded
if args.cuda and torch.cuda.is_available():
    model.cuda()

total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())

if not args.quiet:
    print("sha: {}".format(get_sha().strip()))
    print('args:', args)
    print('model total parameters:', total_params)
    print('model architecture:')
    print(model)


# Loop over epochs.
args.anneal = 0.01
lr = args.lr
best_val_loss = 1e10
true_marginal = 0
stored_loss = 100000000


def flush():
    if args.save is not None:
        with open(args.save, 'w') as f:
            torch.save(model.state_dict(), f)
            print('saved parameters to {}'.format(args.save))
    if args.save_params is not None:
        with open(args.save_params, 'wb') as f:
            # NOTE: these are all in log-space
            T = model.T.data.cpu()
            pi = model.pi.data.cpu()
            emit = model.emit.data.cpu()
            if hasattr(model, 'hidden'):
                hidden = model.hidden.data.cpu()
            else:
                hidden = None
            torch.save((T, pi, emit, hidden), f)


# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, verbose=True, threshold=0.01, min_lr=1e-5)

    val_loss, val_nll, true_marginal = model.evaluate(val_loader, args, args.num_importance_samples)
    print("-" * 89)
    print("ELBO: {:5.2f}, val_nll: {:5.2f}, ELBO ppl: {:5.2f}".format(val_loss, val_nll, np.exp(val_loss)))
    print("-" * 89)

    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()

        if epoch < args.kl_anneal_delay:
            args.anneal = args.kl_anneal_start

        train_loss = model.train_epoch(train_loader, optimizer, epoch, args, args.num_importance_samples)

        # let's ignore ASGD for now
        val_loss, val_nll, true_marginal = model.evaluate(val_loader, args, args.num_importance_samples)
        print("anneal: {:.3f}".format(args.anneal))
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            flush()
        if not args.quiet:
            if val_loss < 10:
                ppl = np.exp(val_loss)
            else:
                ppl = np.inf

            print('-' * 80)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid ELBO {:5.2f} | valid NLL {:5.2f} | PPL: {:5.2f}'
                  ''.format(epoch, (time.time() - epoch_start_time), val_loss, val_nll, ppl))
            print('-' * 80)

        if epoch % 10 == 0:
            args.lr = args.lr * 0.8
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
except KeyboardInterrupt:
    if not args.quiet:
        print('-' * 89)
        print('Exiting from training early')
if args.print_best:
    print("{},{}".format(-best_val_loss, true_marginal))
