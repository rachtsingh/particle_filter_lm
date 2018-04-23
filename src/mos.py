"""
Things that have been dropped that should be reused:
    1. WeightDrop and embedded_dropout
"""

import torch
import torch.nn as nn
from torch.nn import functional as F  # noqa: F401
from torch.autograd import Variable
from torch.distributions import RelaxedOneHotCategorical, OneHotCategorical  # noqa: F401

import pdb  # noqa: F401

from src.utils import VERSION
# from embed_regularize import embedded_dropout
from src.locked_dropout import LockedDropout
# from weight_drop import WeightDrop


class MFVI_Mos(nn.Module):
    """
    This module uses exact MFVI to optimize the generative model of:
    https://github.com/zihangdai/mos
    Significant portions of that code is used.
    """

    def __init__(self, ntoken, word_dim, nhid_inf, nhid, nhidlast, nlayers=1,
                 dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0,
                 tie_weights=False, ldropout=0.5, n_experts=10):
        super(MFVI_Mos, self).__init__()
        self.lockdrop = LockedDropout()

        # encoder / inference net
        self.inp_embedding = nn.Embedding(ntoken, word_dim)
        self.encoder = torch.nn.LSTM(word_dim, nhid_inf, 1, dropout=0, bidirectional=True)
        self.logits = nn.Sequential(nn.Linear(2 * nhid_inf, nhid_inf),
                                    nn.ReLU(),
                                    nn.Linear(nhid_inf, n_experts))
        self.enc = nn.ModuleList([self.inp_embedding, self.encoder, self.logits])

        # decoder / generative model
        self.rnns = [torch.nn.LSTM(word_dim if l == 0 else nhid, nhid if l != nlayers - 1 else nhidlast, 1, dropout=0) for l in range(nlayers)]
        self.rnns = torch.nn.ModuleList(self.rnns)

        self.prior = nn.Linear(nhidlast, n_experts, bias=False)
        self.latent = nn.Sequential(nn.Linear(nhidlast, n_experts*word_dim), nn.Tanh())
        self.decoder = nn.Linear(word_dim, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462

        if tie_weights:
            self.decoder.weight = self.inp_embedding.weight

        self.init_weights()

        self.word_dim = word_dim
        self.nhid_inf = nhid_inf
        self.nhid = nhid
        self.nhidlast = nhidlast
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.ldropout = ldropout
        self.dropoutl = ldropout
        self.n_experts = n_experts
        self.ntoken = ntoken

        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('param size: {}'.format(size))

    def init_weights(self):
        initrange = 0.1
        self.inp_embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, args, n_particles, test=False):
        if not test:
            n_particles = 1
        seq_len, batch_size = input.size()

        # emb = embedded_dropout(self.inp_embedding, input, dropout=self.dropoute if self.training else 0)
        emb = self.inp_embedding(input)
        hidden = self.init_other_hidden(batch_size, self.nhid_inf, 2)  # bidirectional
        hidden_states, (_, _) = self.encoder(emb, hidden)
        hidden_states = hidden_states.repeat(1, n_particles, 1)  # [seq_len, batch_size, embedding]
        q_logit = self.logits(hidden_states.view(-1, 2 * self.nhid_inf))
        q = nn.functional.softmax(q_logit)

        # generation time
        emb = self.lockdrop(emb, self.dropouti)
        hidden = self.init_hidden(batch_size)
        raw_output = emb
        new_hidden = []

        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)

        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        latent = self.latent(output)
        latent = self.lockdrop(latent, self.dropoutl)
        x_logit = self.decoder(latent.view(-1, self.word_dim))

        prior_logit = self.prior(output).contiguous().view(-1, self.n_experts)
        prior = nn.functional.softmax(prior_logit)

        log_prob = nn.functional.log_softmax(x_logit.view(-1, self.ntoken)).view(-1, self.n_experts, self.ntoken)
        x_given_z = (log_prob * q.unsqueeze(2).expand_as(log_prob)).sum(1)
        
        NLL = nn.NLLLoss(reduce=False)(x_given_z, input.view(-1))
        KL = 
        

        pdb.set_trace()

        return model_output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [(Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast).zero_()),
                 Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast).zero_()))
                for l in range(self.nlayers)]

    def init_other_hidden(self, bsz, dim, directions=1, squeeze=False):
        weight = next(self.parameters()).data
        if squeeze:
            return Variable(weight.new(directions, bsz, dim).zero_().squeeze())
        else:
            return (Variable(weight.new(directions, bsz, dim).zero_()),
                    Variable(weight.new(directions, bsz, dim).zero_()))

    def evaluate(self, data_source, args, num_samples=None):
        self.eval()
        total_loss = 0
        total_nll = 0
        total_tokens = 0

        for batch in data_source:
            if args.cuda:
                batch = batch.cuda()
            data = Variable(batch.squeeze(0).t().contiguous())  # squeeze for 1 billion
            total_tokens += (data.size()[0] * data.size()[1])
            loss, nll = self.forward(data, args, 1, test=True)
            loss = loss.sum()
            total_loss += loss.detach().data
            total_nll += nll

        if args.dataset not in ('1billion', 'ptb'):
            # we don't want to average in the synthetic dataset case
            total_tokens = 1

        if VERSION[1]:
            total_loss = total_loss.item()
        else:
            total_loss = total_loss[0]

        return total_loss / float(total_tokens), total_nll / float(total_tokens)

    def train_epoch(self, train_data, optimizer, epoch, args, num_samples=None):
        self.train()
        total_loss = 0

        for i, batch in enumerate(train_data):
            if args.cuda:
                batch = batch.cuda()
            data = Variable(batch.squeeze(0).t().contiguous())  # squeeze for 1 billion
            optimizer.zero_grad()
            loss, nll = self.forward(data, args, num_samples, test=False)
            tokens = (data.size()[0] * data.size()[1])
            loss = loss.sum() / tokens
            loss.backward()
            optimizer.step()
            total_loss += loss.detach()
            if (i + 1) % args.log_interval == 0:
                print("total loss: {:.3f}, nll: {:.3f}".format(loss.data[0], nll / tokens))

        # actually ignored
        return total_loss.data[0]


class VRNN_MoS_Concrete(nn.Module):
    """
    This is the same generative model (and inference net) as above, but since we
    need to use
    """
    pass
