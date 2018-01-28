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


class PFLM(nn.Module):
    """
    Here, encoder refers to the encoding RNN, unlike in baseline.py
    We're using a single layer RNN on the encoder side, but the decoder is essentially two RNNs
    """

    def __init__(self, ntoken, ninp, nhid, z_dim, nlayers, dropout=0.5, dropouth=0.5,
                 dropouti=0.5, dropoute=0.1, wdrop=0., autoregressive_prior=False):
        super(PFLM, self).__init__()
        
        # encoder side
        self.inp_embedding = nn.Embedding(ntoken, ninp)
        self.encoder = torch.nn.LSTM(ninp, nhid, 1, dropout=0, bidirectional=True)
        self.enc = nn.ModuleList([self.inp_embedding, self.encoder])

        # latent stuff
        self.mean = nn.Linear(z_dim, z_dim)
        self.logvar = nn.Linear(z_dim, z_dim)

        if autoregressive_prior:
            self.ar_prior_mean = nn.LSTMCell(z_dim, z_dim)

        # decoder side
        self.z_decoder = nn.LSTMCell(2 * nhid, z_dim) # we're going to feed the output from the last step in at every input
        self.dec_embedding = nn.Embedding(ntoken, ninp)
        self.dropout = nn.Dropout(dropout)
        self.decoder = torch.nn.LSTM(z_dim + ninp, nhid, 1, dropout=0)

        self.out_embedding = nn.Linear(nhid, ntoken)
        self.dec = nn.ModuleList([self.z_decoder, self.dec_embedding, self.decoder, self.out_embedding])

        self.init_weights()

        self.z_dim = z_dim
        self.ninp = ninp
        self.nhid = nhid
        self.ntoken = ntoken
        self.nlayers = nlayers
        self.dropoutp = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute

        self.aprior = autoregressive_prior

    def init_weights(self):
        initrange = 0.1
        self.inp_embedding.weight.data.uniform_(-initrange, initrange)
        self.out_embedding.bias.data.fill_(0)
        self.out_embedding.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz, dim, directions=1):
        weight = next(self.parameters()).data
        return (Variable(weight.new(directions, bsz, dim).zero_()),
                Variable(weight.new(directions, bsz, dim).zero_()))

    def forward(self, input, targets, args, return_h=False):
        """
        input: [seq len x batch]
        """
        seq_len, batch_sz = input.size()
        emb = self.inp_embedding(input)
        hidden = self.init_hidden(batch_sz, self.nhid, 2) # bidirectional
        hidden_states, (h, c) = self.encoder(emb, hidden)

        # run the z-decoder at this point
        z_samples = []
        means = []
        logvars = []
        if self.aprior:
            prior_means = []
            p_h, p_c = self.init_hidden(batch_sz, self.z_dim) # initially zero
        h, c = self.init_hidden(batch_sz, self.z_dim)
        h = h.squeeze(0)
        c = c.squeeze(0)
        for i in range(seq_len):
            h, c = self.z_decoder(hidden_states[i], (h, c))
            mean, logvar = self.mean(h), self.logvar(h)

            # add the offset prior mean before mutation
            if self.aprior:
                prior_means.append(p_h)
            
            # build the next z sample
            std = (logvar/2).exp()
            eps = Variable(torch.randn(mean.size()))
            if torch.cuda.is_available():
                eps = eps.cuda()
            h = (eps * std) + mean
           
            if self.aprior:
                # build the next mean prediction, feeding in this z
                p_h, p_c = self.ar_prior_mean(h, (p_h, p_c))

            z_samples.append(h)
            means.append(mean)
            logvars.append(logvar)

        # now we pass this through the decoder, also adding the source sentence offset
        # targets has <bos> and doesn't have <eos>
        # also, we weaken decoder by removing some words
        out_emb = self.dropout(self.dec_embedding(targets))
        samples = torch.stack(z_samples)
        decoder_input = torch.cat([out_emb, samples], 2)

        hidden = self.init_hidden(batch_sz, self.nhid)
        raw_out, _ = self.decoder(decoder_input, hidden)
        seq_len, batches, nhid = raw_out.size()
        resized = raw_out.view(seq_len * batches, nhid)
        logits = self.out_embedding(resized).view(seq_len, batches, self.ntoken)

        if self.aprior:
            return logits, means, logvars, prior_means
        else:
            return logits, means, logvars

    def elbo(self, logits, targets, criterion, means, logvars, args, prior_means=None):
        seq_len, batch_size, ntokens = logits.size()
        NLL = seq_len * batch_size * criterion(logits.view(-1, ntokens), targets.view(-1))
        KL = 0
        if prior_means is None:
            for mean, logvar in zip(means, logvars):
                KL += -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        else:
            for mean, prior_mean, logvar in zip(means, prior_means, logvars):
                KL += -0.5 * torch.sum(1 + logvar - (mean - prior_mean).pow(2) - logvar.exp())
        return (NLL + args.anneal * KL), NLL, KL, seq_len * batch_size

    def evaluate(self, corpus, data_source, args, criterion):
        self.eval()
        args.anneal = 1.
        total_loss = 0
        total_nll = 0
        total_tokens = 0
        for batch in data_source:
            data, targets = batch.text, batch.target
            logits, means, logvars = self.forward(data, targets, args)
            loss, NLL, KL, tokens = self.elbo(logits, data, criterion, means, logvars, args)
            total_loss += loss.detach().data
            total_nll += NLL.detach().data
            total_tokens += tokens
        print("eval: {:.2f} NLL".format(total_nll[0] / total_loss[0]))
        return total_loss[0] / total_tokens, total_nll[0] / total_tokens

    def train_epoch(self, corpus, train_data, criterion, optimizer, epoch, args):
        self.train()
        dataset_size = len(train_data.data())  # this will be approximate
        total_loss = 0
        total_tokens = 0
        batch_idx = 0
        for batch in train_data:
            if epoch > args.kl_anneal_delay:
                args.anneal += args.kl_anneal_rate
            optimizer.zero_grad()
            data, targets = batch.text, batch.target
            if self.aprior:
                logits, means, logvars, prior_means = self.forward(data, targets, args)
                elbo, NLL, KL, tokens = self.elbo(logits, data, criterion, means, logvars, args, prior_means=prior_means)
            else:
                logits, means, logvars = self.forward(data, targets, args)
                elbo, NLL, KL, tokens = self.elbo(logits, data, criterion, means, logvars, args)
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
