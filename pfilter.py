"""
This model is the particle filter LM I've been building up towards. In particular it needs a slightly different
architecture because we need to evaluate log-likelihood inside the forward pass
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import softmax
import math
from tqdm import tqdm
import pdb  # noqa: F401

from utils import print_in_epoch_summary, log_sum_exp
from locked_dropout import LockedDropout  # noqa: F401
from embed_regularize import embedded_dropout  # noqa: F401

LOG_2PI = math.log(2 * math.pi)


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
        self.z_decoder = nn.LSTMCell(2 * nhid, z_dim)  # we're going to feed the output from the last step in at every input
        self.dec_embedding = nn.Embedding(ntoken, ninp)
        self.dropout = nn.Dropout(dropout)
        self.decoder = torch.nn.LSTMCell(z_dim + ninp, nhid)  # note the difference here

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

    def init_hidden(self, bsz, dim, directions=1, squeeze=False):
        weight = next(self.parameters()).data
        if squeeze:
            return (Variable(weight.new(directions, bsz, dim).zero_().squeeze()),
                    Variable(weight.new(directions, bsz, dim).zero_().squeeze()))
        else:
            return (Variable(weight.new(directions, bsz, dim).zero_()),
                    Variable(weight.new(directions, bsz, dim).zero_()))

    def forward(self, input, targets, args, n_particles, criterion):
        """
        This version takes the inputs, and does not expose the logits, but instead
        computes the losses directly
        """

        # run the input and teacher-forcing inputs through the embedding layers here
        seq_len, batch_sz = input.size()
        emb = self.inp_embedding(input)
        hidden = self.init_hidden(batch_sz, self.nhid, 2)  # bidirectional
        hidden_states, (h, c) = self.encoder(emb, hidden)

        # teacher-forcing
        out_emb = self.dropout(self.dec_embedding(targets))

        # now, we'll replicate it for each particle - it's currently [seq_len x batch_sz x nhid]
        hidden_states = Variable(torch.arange(batch_sz).view(1, batch_sz, 1).cuda().repeat(seq_len, 1, self.nhid * 2))
        out_emb = Variable(torch.arange(batch_sz).view(1, batch_sz, 1).cuda().repeat(seq_len, 1, self.ninp))
        hidden_states = hidden_states.repeat(1, n_particles, 1)
        out_emb = out_emb.repeat(1, n_particles, 1)
        # now [seq_len x (n_particles x batch_sz) x nhid]

        # out_emb, hidden_states should be viewed as (n_particles x batch_sz) - this means that's true for h as well

        # run the z-decoder at this point, evaluating the NLL at each step
        p_h, p_c = self.init_hidden(batch_sz * n_particles, self.z_dim, squeeze=True)  # initially zero

        h, c = self.init_hidden(batch_sz * n_particles, self.z_dim, squeeze=True)
        d_h, d_c = self.init_hidden(batch_sz * n_particles, self.nhid, squeeze=True)

        # we maintain two Tensors, [seq_len x batch_sz]:
        ancestors = input.data.new(seq_len + 1, n_particles, batch_sz).long().zero_()
        init_ancestors = torch.arange(n_particles).view(-1, 1).repeat(1, batch_sz).long()
        if d_h.is_cuda:
            init_ancestors = init_ancestors.cuda()
        ancestors[0] = init_ancestors

        weights = Variable(hidden_states.data.new(seq_len, batch_sz * n_particles))
        nlls = hidden_states.data.new(seq_len, batch_sz * n_particles)

        for i in range(seq_len):
            h, c = self.z_decoder(hidden_states[i], (h, c))
            mean, logvar = self.mean(h), self.logvar(h)

            # build the next z sample
            std = (logvar/2).exp()
            eps = Variable(torch.randn(mean.size()))
            if torch.cuda.is_available():
                eps = eps.cuda()
            z = (eps * std) + mean
            h = z

            # now, compute the log-likelihood of the data given this mean, and the input out_emb
            d_h, d_c = self.decoder(torch.cat([z, out_emb[i]], 1), (d_h, d_c))
            logits = self.out_embedding(d_h)
            NLL = criterion(logits, input[i].repeat(n_particles))
            nlls[i] = NLL.data

            # compute the weight using `reweight` on page (4)
            f_term = -0.5 * (LOG_2PI) - 0.5 * (z - p_h).pow(2).sum(1)  # prior
            r_term = -0.5 * (LOG_2PI + 2 * logvar).sum(1) - 0.5 * eps.pow(2).sum(1)  # proposal
            weights[i] = -NLL + f_term - r_term

            # sample ancestors, and reindex everything
            probs = softmax(weights[i].view(n_particles, batch_sz), dim=0).data
            probs += 0.01
            probs = probs / probs.sum(0, keepdim=True)
            ancestors[i] = torch.multinomial(probs.transpose(0, 1), n_particles, True)

            # now, reindex h + c, which is the most important thing
            offsets = n_particles * torch.arange(batch_sz).unsqueeze(1).repeat(1, n_particles).long()
            if ancestors[i].is_cuda:
                offsets = offsets.cuda()
            unrolled_idx = (ancestors[i].t().contiguous()+offsets).view(-1)
            h = h[unrolled_idx]
            c = c[unrolled_idx]
            p_h = p_h[unrolled_idx]
            p_c = p_c[unrolled_idx]
            d_h = d_h[unrolled_idx]
            d_c = d_c[unrolled_idx]

            # build the next mean prediction, feeding in the correct ancestor
            p_h, p_c = self.ar_prior_mean(h, (p_h, p_c))

        # now, we calculate the final log-marginal estimator
        fivo_loss = -(log_sum_exp(weights.view(seq_len, n_particles, batch_sz), 1) - math.log(n_particles)).sum()
        nll = nlls.view(seq_len, n_particles, batch_sz).mean(-1).sum()
        return fivo_loss, nll, (seq_len * batch_sz)

    def evaluate(self, corpus, data_source, args, criterion, iwae=False, num_importance_samples=3):
        self.eval()
        old_anneal = args.anneal  # save to replace after evaluation
        args.anneal = 1.
        total_loss = 0
        total_nll = 0
        total_tokens = 0
        for batch in data_source:
            data, targets = batch.text, batch.target
            elbo, NLL, tokens = self.forward(data, targets, args, num_importance_samples, criterion)
            total_loss += elbo.detach().data
            total_nll += NLL
            total_tokens += tokens
        print("no annealing yet")
        print("eval: {:.2f} NLL".format(total_nll / total_loss[0]))
        args.anneal = old_anneal

        # duplicate total_loss because we don't have a separate ELBO loss here, though we can grab it
        return total_loss[0] / total_tokens, total_loss[0] / total_tokens, total_nll / total_tokens

    def train_epoch(self, corpus, train_data, criterion, optimizer, epoch, args, num_importance_samples):
        self.train()
        dataset_size = len(train_data.data())  # this will be approximate
        total_loss = 0
        total_tokens = 0
        batch_idx = 0
        print("batch_sz", args.batch_size)
        for batch in tqdm(train_data):
            if epoch > args.kl_anneal_delay:
                args.anneal = min(args.anneal + args.kl_anneal_rate, 1.)
            optimizer.zero_grad()
            data, targets = batch.text, batch.target
            elbo, NLL, tokens = self.forward(data, targets, args, num_importance_samples, criterion)
            loss = elbo/tokens
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.parameters(), args.clip)
            optimizer.step()

            total_loss += elbo.detach()
            total_tokens += tokens

            # print if necessary
            if batch_idx % args.log_interval == 0 and batch_idx > 0:
                print_in_epoch_summary(epoch, batch_idx, args.batch_size, dataset_size,
                                       loss.data[0], NLL / tokens, {},
                                       tokens, "anneal={:.2f}".format(args.anneal))
            batch_idx += 1  # because no cheap generator smh
        return total_loss[0] / total_tokens
