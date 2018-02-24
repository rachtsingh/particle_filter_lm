"""
This is a reimplementation of Bowman et. al.'s Generating Sentences from a Continuous Space
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb  # noqa: F401

from utils import print_in_epoch_summary
from locked_dropout import LockedDropout  # noqa: F401
from embed_regularize import embedded_dropout  # noqa: F401


class RVAE(nn.Module):
    """
    Here, encoder refers to the encoding RNN, unlike in baseline.py
    We're using a single layer RNN on both the encoder and decoder side, and removing the option to tie the encoder and decoder embeddings together
    """

    def __init__(self, ntoken, ninp, nhid, z_dim, nlayers, dropout=0.5, dropouth=0.5,
                 dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=True):
        super(RVAE, self).__init__()

        # input side
        self.inp_embedding = nn.Embedding(ntoken, ninp)
        self.encoder = torch.nn.LSTM(ninp, nhid, 1, dropout=0)
        self.enc = nn.ModuleList([self.inp_embedding, self.encoder])

        # latent
        self.mean = nn.Linear(nhid, z_dim)
        self.logvar = nn.Linear(nhid, z_dim)
        # self.prior_mean = nn.Parameter(torch.zeros(z_dim))
        # self.prior_logvar = nn.Parameter(torch.zeros(z_dim))
        self.latent = nn.ModuleList([self.mean, self.logvar])

        # decoder side
        # self.lockdrop = LockedDropout()
        self.dropout = nn.Dropout(dropout)
        self.dec_embedding = nn.Embedding(ntoken, ninp)
        self.decoder = torch.nn.LSTM(ninp + z_dim, nhid, 1, dropout=0)
        self.out_embedding = nn.Linear(nhid, ntoken)
        self.latent_linear = nn.Linear(z_dim, nhid)
        self.dec = nn.ModuleList([self.dropout, self.dec_embedding, self.latent_linear, self.decoder, self.out_embedding])

        self.init_weights()

        self.ninp = ninp
        self.nhid = nhid
        self.ntoken = ntoken
        self.nlayers = nlayers
        self.dropoutp = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def init_weights(self):
        pass
        # initrange = 0.1
        # self.inp_embedding.weight.data.uniform_(-initrange, initrange)
        # self.dec_embedding.weight.data.uniform_(-initrange, initrange)
        # self.out_embedding.bias.data.fill_(0)
        # self.out_embedding.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz, dim):
        weight = next(self.parameters()).data
        return (Variable(weight.new(1, bsz, dim).zero_()),
                Variable(weight.new(1, bsz, dim).zero_()))

    def forward(self, input, targets, args, return_h=False):
        """
        input: [seq len x batch]
        """
        seq_len, batch_sz = input.size()
        emb = self.inp_embedding(input)
        hidden = self.init_hidden(batch_sz, self.nhid)
        _, (h, c) = self.encoder(emb, hidden)

        # now I have a [sentence size x batch x nhid] tensor in output
        last_output = h[-1]
        mean = self.mean(last_output)
        logvar = self.logvar(last_output)
        std = (logvar/2).exp()
        eps = torch.randn(mean.size())
        if torch.cuda.is_available():
            eps = eps.cuda()
        qz = (Variable(eps) * std) + mean
        samples = qz.unsqueeze(0).expand(seq_len, batch_sz, qz.size(1))

        # now we pass this through the decoder, also adding the source sentence offset by 1 (targets)
        # targets has <bos> and doesn't have <eos>
        # also, we weaken decoder by removing some words
        out_emb = self.dropout(self.dec_embedding(targets))
        decoder_input = torch.cat([out_emb, samples], 2)
        hidden = self.init_hidden(batch_sz, self.nhid)
        # hidden[0][-1] = self.latent_linear(qz)

        raw_out, _ = self.decoder(decoder_input, hidden)
        seq_len, batches, nhid = raw_out.size()
        resized = raw_out.view(seq_len * batches, nhid)
        logits = self.out_embedding(resized).view(seq_len, batches, self.ntoken)

        return logits, mean, logvar

    def elbo(self, logits, targets, criterion, mean, logvar, args, anneal, prior_mean=None, prior_logvar=None):
        # t = mean.data.__class__
        # if prior_mean is None:
        #     prior_mean = Variable(torch.zeros(mean.size()).type(t))
        # else:
        #     prior_mean = prior_mean.unsqueeze(0).expand_as(mean)
        # if prior_logvar is None:
        #     prior_logvar = Variable(torch.zeros(logvar.size()).type(t))
        # else:
        #     prior_logvar = prior_logvar.unsqueeze(0).expand_as(logvar)
        seq_len, batch_size, ntokens = logits.size()
        NLL = seq_len * batch_size * criterion(logits.view(-1, ntokens), targets.view(-1))
        # KL = 0.5 * torch.sum((logvar.exp() + (mean - prior_mean).pow(2))/prior_logvar.exp() - 1. + prior_logvar - logvar)
        KL = -0.5 * torch.sum(logvar - torch.pow(mean, 2) - torch.exp(logvar) + 1)
        return (NLL + anneal * KL), NLL, KL, seq_len * batch_size

    def evaluate(self, corpus, data_source, args, criterion):
        self.eval()
        total_loss = 0
        total_nll = 0
        total_tokens = 0
        for batch in data_source:
            data, targets = batch.text, batch.target
            logits, mean, logvar = self.forward(data, targets, args)
            loss, NLL, KL, tokens = self.elbo(logits, data, criterion, mean, logvar, args, anneal=1.)
            total_loss += loss.detach().data
            total_nll += NLL.detach().data
            total_tokens += tokens
        print("eval: {:.3f} NLL | current anneal: {:.3f}".format(total_nll[0] / total_loss[0], args.anneal))
        return total_loss[0] / total_tokens, total_nll[0] / total_tokens

    def train_epoch(self, corpus, train_data, criterion, optimizer, epoch, args):
        self.train()
        dataset_size = len(train_data.data())  # this will be approximate
        total_loss = 0
        total_tokens = 0
        batch_idx = 0
        for batch in train_data:
            if epoch > args.kl_anneal_delay and args.anneal < 1.:
                args.anneal += args.kl_anneal_rate
            optimizer.zero_grad()
            data, targets = batch.text, batch.target
            logits, mean, logvar = self.forward(data, targets, args)
            elbo, NLL, KL, tokens = self.elbo(logits, data, criterion, mean, logvar, args, args.anneal)
            loss = elbo/tokens
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.parameters(), args.clip)
            optimizer.step()

            total_loss += elbo.detach()
            total_tokens += tokens

            # print if necessary
            if batch_idx % args.log_interval == 0 and batch_idx > 0:
                print_in_epoch_summary(epoch, batch_idx, args.batch_size, dataset_size,
                                       loss.data[0], NLL.data[0] / tokens, {'normal': KL.data[0] / tokens},
                                       tokens, "anneal={:.2f}".format(args.anneal))
            batch_idx += 1  # because no cheap generator smh
        return total_loss[0] / total_tokens
