"""
This is the baseline model, which uses a RNN-LM directly.
"""
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import repackage_hidden
import numpy as np

from locked_dropout import LockedDropout
from embed_regularize import embedded_dropout


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5,
                 dropoute=0.1, wdrop=0, tie_weights=False):
        super(RNNModel, self).__init__()
        # variational dropout
        self.lockdrop = LockedDropout()
        # self.idrop = nn.Dropout(dropouti)
        # self.hdrop = nn.Dropout(dropouth)
        # self.edrop = nn.Dropout(dropoute)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnns = [torch.nn.LSTM(ninp if layer == 0 else nhid,
                                   nhid if layer != nlayers - 1 else (ninp if tie_weights else nhid),
                                   1, dropout=0) for layer in range(nlayers)]
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []

        # this is multilayer because the Salesforce version is, and I'll need it later.
        # but for now, all experiments will be with 1-layer versions
        raw_outputs = []
        outputs = []
        for layer, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[layer])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if layer != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        result = decoded.view(output.size(0), output.size(1), decoded.size(1))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data

        def chosen_size(l):
            return self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)
        return [(Variable(weight.new(1, bsz, chosen_size(l)).zero_()),
                Variable(weight.new(1, bsz, chosen_size(l)).zero_()))
                for l in range(self.nlayers)]

    def evaluate(self, corpus, data_source, args, criterion, iwae=False, num_importance_samples=None):
        """
        IWAE metrics are disabled for this model
        """
        # Turn on evaluation mode which disables dropout.
        self.eval()
        total_loss = 0
        ntokens = len(corpus)
        hidden = self.init_hidden(args.batch_size)
        for batch in data_source:
            data, targets = batch.text, batch.target
            output, hidden = self.forward(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * args.batch_size * criterion(output_flat, targets.view(-1)).data
            hidden = repackage_hidden(hidden)
        return total_loss[0] / len(data_source.dataset[0].text)

    def train_epoch(self, corpus, train_data, criterion, optimizer, epoch, args):
        # Turn on training mode which enables dropout.
        total_loss = 0
        start_time = time.time()
        ntokens = len(corpus)
        hidden = self.init_hidden(args.batch_size)
        for batch_num, batch in enumerate(train_data):
            data, targets = batch.text, batch.target

            lr2 = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = lr2 * batch.text.size(0) / args.bptt
            self.train()

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = repackage_hidden(hidden)
            optimizer.zero_grad()

            output, hidden, rnn_hs, dropped_rnn_hs = self.forward(data, hidden, return_h=True)
            raw_loss = criterion(output.view(-1, ntokens), targets.view(-1))

            loss = raw_loss

            # Activation Regularization
            loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])

            # Temporal Activation Regularization (slowness)
            loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm(self.parameters(), args.clip)
            optimizer.step()

            total_loss += raw_loss.data
            optimizer.param_groups[0]['lr'] = lr2
            if batch_num % args.log_interval == 0 and batch_num > 0:
                cur_loss = total_loss[0] / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch_num, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                        elapsed * 1000 / args.log_interval, cur_loss, np.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
