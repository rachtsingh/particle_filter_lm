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
    Note that if tie_weights = True, then ninp = nhid
    We're using a single layer RNN on both the encoder and decoder side
    """

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5,
                 dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=True):
        super(RVAE, self).__init__()
        self.lockdrop = LockedDropout()
        self.inp_embedding = nn.Embedding(ntoken, ninp)

        # it's not a complicated model, these are the main parameters
        self.encoder = torch.nn.LSTM(ninp, nhid, 1, dropout=0)
        self.mean = nn.Linear(nhid, 2 * nhid)
        self.logvar = nn.Linear(nhid, 2 * nhid)
        self.decoder = torch.nn.LSTM(ninp, nhid, 1, dropout=0)

        self.out_embedding = nn.Linear(nhid, ntoken)
        if tie_weights:
            assert nhid == ninp, "nhidden has to equal ninp for tying"
            self.out_embedding.weight = self.inp_embedding.weight

        self.init_weights()

        self.ninp = ninp
        self.nhid = nhid
        self.ntoken = ntoken
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def init_weights(self):
        initrange = 0.1
        self.inp_embedding.weight.data.uniform_(-initrange, initrange)
        self.out_embedding.bias.data.fill_(0)
        self.out_embedding.weight.data.uniform_(-initrange, initrange)

    def init_hidden_encoder(self, bsz):
        weight = next(self.parameters()).data
        chosen_size = self.nhid
        return (Variable(weight.new(1, bsz, chosen_size).zero_()),
                Variable(weight.new(1, bsz, chosen_size).zero_()))

    def forward(self, input, hidden, targets, return_h=False):
        """
        input: [seq len x batch x V]
        """
        # emb = embedded_dropout(self.inp_embedding, input, dropout=self.dropoute if self.training else 0)
        emb = self.inp_embedding(input)
        # emb = self.lockdrop(emb, self.dropouti)
        raw_output, new_h = self.encoder(emb, hidden)
        # output = self.lockdrop(raw_output, self.dropout)
        output = raw_output

        # now I have a [sentence size x batch x nhid] tensor in output
        last_output = output[-1]
        mean = self.mean(last_output)
        logvar = self.logvar(last_output)
        std = (logvar/2).exp()
        eps = torch.randn(mean.size())
        if torch.cuda.is_available():
            eps = eps.cuda()
        samples = (Variable(eps) * std) + mean
        output_hidden = torch.chunk(samples, 2, 1)

        # now we pass this through the decoder
        # targets has <bos> and doesn't have <eos>
        out_emb = self.inp_embedding(targets)
        raw_out, _ = self.decoder(out_emb, output_hidden)
        seq_len, batches, nhid = raw_out.size()
        decoder_output = self.out_embedding(raw_out.view(seq_len * batches, nhid))
        logits = decoder_output.view(seq_len, batches, self.ntoken)

        return logits, mean, logvar

    def elbo(self, logits, targets, criterion, mean, logvar, anneal, args):
        seq_len, batch_size, ntokens = logits.size()
        NLL = seq_len * batch_size * criterion(logits.view(-1, ntokens), targets.view(-1))
        KL = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return (NLL + anneal * KL), NLL, KL, seq_len * batch_size

    def evaluate(self, corpus, data_source, args, criterion):
        self.eval()
        total_loss = 0
        total_tokens = 0
        for batch in data_source:
            hidden = self.init_hidden_encoder(batch.batch_size)
            data, targets = batch.text, batch.target
            logits, mean, logvar = self.forward(data, hidden, targets)
            loss, tokens = self.elbo(logits, targets, criterion, mean, logvar, 1, args)
            total_loss += loss
            total_tokens += tokens
        return total_loss[0] / total_tokens

    def train_epoch(self, corpus, train_data, criterion, optimizer, epoch, args):
        self.train()
        dataset_size = len(train_data.data())  # this will be approximate
        total_loss = 0
        total_tokens = 0
        batch_idx = 0
        for batch in train_data:
            hidden = self.init_hidden_encoder(batch.batch_size)
            optimizer.zero_grad()
            data, targets = batch.text, batch.target
            logits, mean, logvar = self.forward(data, hidden, targets)
            elbo, NLL, KL, tokens = self.elbo(logits, targets, criterion, mean, logvar, 1, args)
            loss = elbo/tokens
            loss.backward()

            total_loss += elbo
            total_tokens += tokens

            # print if necessary
            if batch_idx % args.log_interval == 0 and batch_idx > 0:
                print_in_epoch_summary(epoch, batch_idx, args.batch_size, dataset_size,
                                       loss.data[0], NLL.data[0] / tokens, {'normal': KL.data[0] / tokens},
                                       tokens, "anneal={:.2f}".format(1))
            batch_idx += 1  # because no cheap generator smh
        return total_loss[0] / total_tokens
