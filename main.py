import argparse
import time
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import pdb  # noqa: F401

from torchtext.datasets import PennTreeBank
from data import PTBSeq2Seq
from utils import get_sha

import baseline
import rvae
import sequential
import pfilter

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--dataset', type=str, default='ptb',
                    help='one of [ptb (default), wt2]')
parser.add_argument('--model', type=str, default='rvae',
                    help='type of model to use (baseline, rvae, sequential, filter)')
parser.add_argument('--emsize', type=int, default=512,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1024,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--no-autoregressive-prior', action='store_true',
                    help='disable the autoregressive prior on z for the filter model')
parser.add_argument('--z-dim', type=int, default=32,
                    help='dimensionality of the hidden z')
parser.add_argument('--lr', type=float, default=1.0,
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
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.25,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.1,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--not-tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--max-kl-penalty', type=float, default=0.,
                    help='maximum KL penalty to allow (essentially gradient clips the KL)')
parser.add_argument('--no-iwae', action='store_true',
                    help='whether to disable reporting of the IWAE metric instead of KL')
parser.add_argument('--num-importance-samples', type=int, default=5,
                    help='number of samples to take for IWAE')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=250, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--prof', type=str, default=None,
                    help='If specified, profile the first 10 batches and dump to <prof>')
parser.add_argument('--use-sru', action='store_true',
                    help='whether to use Tao\'s optimized RNN implementation [NOTE: not currently implemented]')
args = parser.parse_args()

print("running {}".format(' '.join(sys.argv)))

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
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

if args.dataset == 'ptb' and args.model == 'baseline':
    train_data, val_data, test_data = PennTreeBank.iters(batch_size=args.batch_size,
                                                         bptt_len=args.bptt,
                                                         device=device)
    corpus = train_data.dataset.fields['text'].vocab
elif args.dataset == 'ptb':
    if args.no_iwae:
        train_data, val_data, test_data = PTBSeq2Seq.iters(batch_size=args.batch_size, device=device)
    else:
        # in IWAE evaluation, we want training to stay fast while eval has smaller batches ( * num_importance_samples)
        small_batch = 8 * (int(args.batch_size / args.num_importance_samples) // 8)
        if args.model == 'sequential':
            train_data, val_data, test_data = PTBSeq2Seq.iters(batch_sizes=(args.batch_size, small_batch, small_batch), device=device)
        elif args.model == 'filter':
            # in the filter, everything is IWAE-fied, essentially
            args.batch_size = small_batch  # so that inside-epoch tracking is correct
            train_data, val_data, test_data = PTBSeq2Seq.iters(batch_size=small_batch, device=device)
        else:
            train_data, val_data, test_data = PTBSeq2Seq.iters(batch_size=args.batch_size, device=device)

    corpus = train_data.dataset.fields['target'].vocab
ntokens = len(corpus)

# BASELINE RNNLM
if args.model == 'baseline':
    model = baseline.RNNModel(ntokens, args.emsize, args.nhid, args.nlayers,
                              args.dropout, args.dropouth, args.dropouti,
                              args.dropoute, args.wdrop, not args.not_tied)

# BASELINE RVAE
elif args.model == 'rvae':
    model = rvae.RVAE(ntokens, args.emsize, args.nhid, args.z_dim, 1, args.dropout, args.dropouth,
                      args.dropouti, args.dropoute, args.wdrop, not args.not_tied)

# OUR 1st MODEL (A SEQUENTIALLY GENERATED VAE)
elif args.model == 'sequential':
    model = sequential.SequentialLM(ntokens, args.emsize, args.nhid, args.z_dim, 1, args.dropout,
                                    args.dropouth, args.dropouti, args.dropoute, args.wdrop,
                                    not args.no_autoregressive_prior)

elif args.model == 'filter':
    model = pfilter.PFLM(ntokens, args.emsize, args.nhid, args.z_dim, 1, args.dropout,
                         args.dropouth, args.dropouti, args.dropoute, args.wdrop,
                         not args.no_autoregressive_prior)

if args.cuda and torch.cuda.is_available():
    model.cuda()
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())

print("sha: {}".format(get_sha().strip()))
print('args:', args)
print('model total parameters:', total_params)
print('model architecture:')
print(model)

criterion = nn.CrossEntropyLoss(reduce=args.no_iwae)

# Loop over epochs.
args.anneal = 0.01
lr = args.lr
best_val_loss = []
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()

        if epoch < args.kl_anneal_delay:
            args.anneal = 0.0001

        if epoch in (15, 25, 35, 45) and args.model == 'rvae':
            args.lr = 0.9 * args.lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        train_loss = model.train_epoch(corpus, train_data, criterion, optimizer, epoch, args, args.num_importance_samples)

        # let's ignore ASGD for now
        val_loss, val_elbo, val_nll = model.evaluate(corpus, val_data, args, criterion, not args.no_iwae, args.num_importance_samples)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid IWAE {:5.2f} | valid ELBO {:5.2f} | valid NLL {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), val_loss, val_elbo, val_nll,
                                         math.exp(val_loss) if val_loss < 10. else float('inf')))
        print('-' * 89)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
# with open(args.save, 'rb') as f:
#     model = torch.load(f)
#
# # Run on test data.
# test_loss = model.evaluate(corpus, test_data, args, criterion)
# print('=' * 89)
# print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
#     test_loss, math.exp(test_loss)))
# print('=' * 89)
