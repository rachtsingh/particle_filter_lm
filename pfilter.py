"""
This model is the particle filter LM I've been building up towards. In particular it needs a slightly different
architecture because we need to evaluate log-likelihood inside the forward pass
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import softmax
import math
import sys
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
            # self.ar_prior_mean = nn.LSTMCell(z_dim + ninp, z_dim)

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
        hidden_states = hidden_states.repeat(1, n_particles, 1)
        out_emb = out_emb.repeat(1, n_particles, 1)
        # now [seq_len x (n_particles x batch_sz) x nhid]
        # out_emb, hidden_states should be viewed as (n_particles x batch_sz) - this means that's true for h as well

        # run the z-decoder at this point, evaluating the NLL at each step
        p_h, p_c = self.init_hidden(batch_sz * n_particles, self.z_dim, squeeze=True)  # initially zero
        h, c = self.init_hidden(batch_sz * n_particles, self.z_dim, squeeze=True)
        d_h, d_c = self.init_hidden(batch_sz * n_particles, self.nhid, squeeze=True)

        # we maintain two Tensors, [seq_len x batch_sz]:
        # ancestors = input.data.new(seq_len + 1, n_particles, batch_sz).long().zero_()
        # init_ancestors = torch.arange(n_particles).view(-1, 1).repeat(1, batch_sz).long()
        # if d_h.is_cuda:
        #     init_ancestors = init_ancestors.cuda()
        # ancestors[0] = init_ancestors

        nlls = hidden_states.data.new(seq_len, batch_sz * n_particles)
        loss = 0

        accumulated_weights = -math.log(n_particles)  # will contain log w_{t - 1}
        resamples = 0

        for i in range(seq_len):
            h, c = self.z_decoder(hidden_states[i], (h, c))
            mean = self.mean(h)

            # build the next z sample
            std = 1
            eps = Variable(torch.randn(mean.size()))
            if torch.cuda.is_available():
                eps = eps.cuda()
            z = (eps * std) + mean

            # now, compute the log-likelihood of the data given this mean, and the input out_emb
            d_h, d_c = self.decoder(torch.cat([z, out_emb[i]], 1), (d_h, d_c))
            logits = self.out_embedding(d_h)
            NLL = criterion(logits, input[i].repeat(n_particles))
            nlls[i] = NLL.data

            # compute the weight using `reweight` on page (4)
            f_term = -0.5 * (LOG_2PI) - 0.5 * (z - p_h).pow(2).sum(1)  # prior
            r_term = -0.5 * (LOG_2PI) - 0.5 * eps.pow(2).sum(1)  # proposal
            alpha = -NLL + args.anneal * (f_term - r_term)

            wa = accumulated_weights + alpha.view(n_particles, batch_sz)

            # sample ancestors, and reindex everything
            Z = log_sum_exp(wa, dim=0)  # line 7
            loss += Z  # line 8
            accumulated_weights = wa - Z  # line 9
            probs = accumulated_weights.data.exp()
            probs += 0.01
            probs = probs / probs.sum(0, keepdim=True)
            effective_sample_size = 1./probs.pow(2).sum(0)

            # resample / RSAMP if 3 batch elements need resampling
            if ((effective_sample_size / n_particles) < 0.6).sum() > 0:
                resamples += 1
                # pdb.set_trace()
                ancestors = torch.multinomial(probs.transpose(0, 1), n_particles, True)

                # now, reindex, which is the most important thing
                offsets = n_particles * torch.arange(batch_sz).unsqueeze(1).repeat(1, n_particles).long()
                if ancestors.is_cuda:
                    offsets = offsets.cuda()
                unrolled_idx = Variable(ancestors.t().contiguous()+offsets).view(-1)
                h = torch.index_select(h, 0, unrolled_idx)
                c = torch.index_select(c, 0, unrolled_idx)
                p_h = torch.index_select(p_h, 0, unrolled_idx)
                p_c = torch.index_select(p_c, 0, unrolled_idx)
                d_h = torch.index_select(d_h, 0, unrolled_idx)
                d_c = torch.index_select(d_c, 0, unrolled_idx)

                # reset accumulated_weights
                accumulated_weights = -math.log(n_particles)  # will contain log w_{t - 1}

            if i != seq_len - 1:
                # build the next mean prediction, feeding in the correct ancestor
                # p_h, p_c = self.ar_prior_mean(torch.cat([h, out_emb[i]], 1), (p_h, p_c))
                p_h, p_c = self.ar_prior_mean(h, (p_h, p_c))

        # now, we calculate the final log-marginal estimator
        nll = nlls.view(seq_len, n_particles, batch_sz).mean(1).sum()
        return -loss.sum(), nll, (seq_len * batch_sz), resamples

    def evaluate(self, corpus, data_source, args, criterion, iwae=False, num_importance_samples=3):
        self.eval()
        old_anneal = args.anneal  # save to replace after evaluation
        args.anneal = 1.
        total_loss = 0
        total_nll = 0
        total_tokens = 0
        total_resamples = 0
        batch_idx = 0
        for batch in tqdm(data_source):
            data, targets = batch.text, batch.target
            elbo, NLL, tokens, resamples = self.forward(data, targets, args, num_importance_samples, criterion)
            total_loss += elbo.detach().data
            total_nll += NLL
            total_tokens += tokens
            total_resamples += resamples
            batch_idx += 1
        print("eval: {:.3f} NLL | current anneal: {:.3f} | average resamples: {:.1f}".format(total_nll / total_loss[0],
                                                                                             old_anneal,
                                                                                             total_resamples/batch_idx))
        args.anneal = old_anneal

        # duplicate total_loss because we don't have a separate ELBO loss here, though we can grab it
        return total_loss[0] / total_tokens, total_loss[0] / total_tokens, total_nll / total_tokens

    def train_epoch(self, corpus, train_data, criterion, optimizer, epoch, args, num_importance_samples):
        self.train()
        dataset_size = len(train_data.data())  # this will be approximate

        def train_loop(profile=False):
            total_loss = 0
            total_tokens = 0
            total_resamples = 0
            batch_idx = 0

            # for pretty printing the loss in each chunk
            last_chunk_loss = 0
            last_chunk_tokens = 0
            last_chunk_resamples = 0

            for batch in tqdm(train_data):
                if profile and batch_idx > 10:
                    print("breaking because profiling finished;")
                    break
                if epoch > args.kl_anneal_delay:
                    args.anneal = min(args.anneal + args.kl_anneal_rate, 1.)
                optimizer.zero_grad()
                data, targets = batch.text, batch.target
                elbo, NLL, tokens, resamples = self.forward(data, targets, args, num_importance_samples, criterion)
                loss = elbo/tokens
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.parameters(), args.clip)
                optimizer.step()

                total_loss += elbo.detach()
                total_tokens += tokens
                total_resamples += resamples

                # print if necessary
                if batch_idx % args.log_interval == 0 and batch_idx > 0:
                    chunk_loss = total_loss.data[0] - last_chunk_loss
                    chunk_tokens = total_tokens - last_chunk_tokens
                    chunk_resamples = (total_resamples - last_chunk_resamples) / args.log_interval
                    print(total_resamples)
                    print_in_epoch_summary(epoch, batch_idx, args.batch_size, dataset_size,
                                           loss.data[0], NLL / tokens,
                                           {'Chunk Loss': chunk_loss / chunk_tokens, 'resamples': chunk_resamples},
                                           tokens, "anneal={:.2f}".format(args.anneal))
                    last_chunk_loss = total_loss.data[0]
                    last_chunk_tokens = total_tokens
                    last_chunk_resamples = total_resamples
                batch_idx += 1  # because no cheap generator smh
            return total_loss, total_tokens

        if args.prof is None:
            total_loss, total_tokens = train_loop(False)
            return total_loss[0] / total_tokens
        else:
            with torch.autograd.profiler.profile() as prof:
                _, _ = train_loop(True)
            prof.export_chrome_trace(args.prof)
            sys.exit(0)
