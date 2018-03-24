"""
Comparison of EM and VI
"""
import argparse
import time
import sys
import numpy as np
import torch
from itertools import product
from torch import nn
import pdb  # noqa: F401

from src.utils import get_sha, VERSION
from src.hmm_dataset import create_hmm_data, HMMData
from src.real_hmm_dataset import OneBillionWord
from src import hmm_filter
from src import hmm

parser = argparse.ArgumentParser(description='Demonstration of Sequential Latent VI for HMMs')
parser.add_argument('--dataset', type=str, default='1billion',
                    help='one of [generate, 1billion, <something>.pt, ...]')
parser.add_argument('--inference', type=str, default='vi',
                    help='which inference method to use (vi, em)')
parser.add_argument('--model', type=str, default='hmm_em',
                    help='which generative model to use (hmm_vi, hmm_em, hmm_deep_em, hmm_deep_vi, '
                                                        'hmm_margin_vi, hmm_mfvi)')
parser.add_argument('--load-hmm', type=str,
                    help='which PyTorch file to load the HMM from, if any')
parser.add_argument('--word-dim', type=int, default=300,
                    help='dimensionality of the word embedding for 1billion')
parser.add_argument('--nhid', type=int, default=32,
                    help='number of hidden units per layer')
parser.add_argument('--z-dim', type=int, default=5,
                    help='dimensionality of the hidden z')
parser.add_argument('--x-dim', type=int, default=10,
                    help='dimensionality of the observed data')
parser.add_argument('--hidden', type=int, default=10,
                    help='dimensionality of hidden size in generative model, if any')
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
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
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
parser.add_argument('--dump-param-traj', type=str, default=None,
                    help='A place to dump out the parameter trajectories as an npz')
parser.add_argument('--embedding', type=str, default=None,
                    help='Which file to load word embeddings from')
parser.add_argument('--load-model', type=str, default=None,
                    help='Which model file to load if any')
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

train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                           shuffle=True, **data_kwargs)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                                         shuffle=True, **data_kwargs)


params = None

# build the model using the true parameters of the generative model
if args.inference == 'vi':
    if args.model == 'hmm_vi':
        model = hmm_filter.HMM_VI(z_dim=args.z_dim, x_dim=args.x_dim, nhid=args.nhid, word_dim=args.word_dim,
                                  temp=args.temp, temp_prior=args.temp_prior, params=None)
    elif args.model == 'hmm_deep_vi':
        model = hmm_filter.HMM_VI_Layers(z_dim=args.z_dim, x_dim=args.x_dim, hidden_size=args.hidden, nhid=args.nhid,
                                         word_dim=args.word_dim, temp=args.temp, temp_prior=args.temp_prior, params=None)
    elif args.model == 'hmm_margin_vi':
        model = hmm_filter.HMM_VI_Marginalized(z_dim=args.z_dim, x_dim=args.x_dim, hidden_size=args.hidden, nhid=args.nhid,
                                               word_dim=args.word_dim, temp=args.temp, temp_prior=args.temp_prior, params=None)
    elif args.model == 'hmm_mfvi':
        model = hmm_filter.HMM_MFVI(z_dim=args.z_dim, x_dim=args.x_dim, hidden_size=args.hidden, nhid=args.nhid,
                                    word_dim=args.word_dim, temp=args.temp, temp_prior=args.temp_prior, params=None)
    elif args.model == 'hmm_mfvi_yoon':
        model = hmm_filter.HMM_MFVI_Yoon(z_dim=args.z_dim, x_dim=args.x_dim, hidden_size=args.hidden, nhid=args.nhid,
                                         word_dim=args.word_dim, temp=args.temp, temp_prior=args.temp_prior, params=None)
    elif args.model == 'hmm_mfvi_yoon_deep':
        model = hmm_filter.HMM_MFVI_Yoon_Deep(z_dim=args.z_dim, x_dim=args.x_dim, hidden_size=args.hidden, nhid=args.nhid,
                                              word_dim=args.word_dim, temp=args.temp, temp_prior=args.temp_prior, params=None)
elif args.inference == 'em':
    if args.model == 'hmm_em':
        model = hmm.HMM_EM(args.z_dim, args.x_dim)
    elif args.model == 'hmm_deep_em':
        model = hmm.HMM_EM_Layers(args.z_dim, args.x_dim, args.hidden)

if args.embedding is not None and model.load_embedding:
    data = torch.load(args.embedding)
    model.load_embedding(data)

# load parameters if we want that comparison:
if args.load_hmm:
    print("loading HMM parameters from {}".format(args.load_hmm))
    params = torch.load(args.load_hmm)
    # just for mixed training comment out again
    params = (params['T'], params['pi'], params['emit'], params['hidden'])
    model.set_params(params)


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
            mode = 'w' if VERSION[0] == 2 else 'wb'
            with open(args.save, mode) as f:
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
    if args.dump_param_traj is not None:
        np.savez(args.dump_param_traj, T=T_traj, pi=pi_traj, emit=emit_traj)

if args.inference == 'vi' and args.load_hmm:
    model.T.requires_grad = False
    model.pi.requires_grad = False
    model.emit.requires_grad = False
    if hasattr(model, 'hidden'):
        model.hidden.requires_grad = False
    print("freezing T, pi, emit, hidden")

if args.load_model:
    model.load_state_dict(torch.load(args.load_model))

# At any point you can hit Ctrl + C to break out of training early.
try:
    inference_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    val_loss, true_marginal = model.evaluate(val_loader, args, args.num_importance_samples)
    print("-" * 89)
    print("-ELBO: {}, ELBO ppl: {}, val before opt: {}".format(val_loss, np.exp(val_loss), np.exp(-true_marginal)))
    print("-" * 89)

    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()

        if epoch < args.kl_anneal_delay:
            args.anneal = args.kl_anneal_start

        # let's only optimize inference in the first few steps
        if args.inference == 'vi' and args.load_hmm:
            if epoch < 100:
                optimizer = inference_optimizer
            elif epoch == 100:
                model.T.requires_grad = True
                model.pi.requires_grad = True
                model.emit.requires_grad = True
                if hasattr(model, 'hidden'):
                    model.hidden.requires_grad = True
                all_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                optimizer = all_optimizer
            else:
                optimizer = all_optimizer
        else:
            optimizer = inference_optimizer  # because that's everything, it's ok here

        train_loss = model.train_epoch(train_loader, optimizer, epoch, args, args.num_importance_samples)

        # let's ignore ASGD for now
        val_loss, true_marginal = model.evaluate(val_loader, args, args.num_importance_samples)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if not args.quiet:
            if val_loss < 10:
                ppl = np.exp(-true_marginal)
            else:
                ppl = np.inf

            print('-' * 80)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid ELBO {:5.2f} | true marginal {:5.2f} | PPL: {:5.2f}'
                  ''.format(epoch, (time.time() - epoch_start_time), val_loss, true_marginal, ppl))
            print('-' * 80)

        if args.dump_param_traj:
            # TODO: update this for new eval_log_marginal / model loading
            T = nn.Softmax(dim=0)(model.T).data.cpu().numpy().T
            pi = nn.Softmax(dim=0)(model.pi).data.cpu().numpy()
            emit = nn.Softmax(dim=0)(model.emit).data.cpu().numpy().T
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
