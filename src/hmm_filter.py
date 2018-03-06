"""
This is an attempt to do inference on HMMs using particle filtering VI
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import RelaxedOneHotCategorical, OneHotCategorical, Categorical
import numpy as np
import math
import sys
import pdb  # noqa: F401

from utils import print_in_epoch_summary, log_sum_exp, any_nans  # noqa: F401
from locked_dropout import LockedDropout  # noqa: F401
from embed_regularize import embedded_dropout  # noqa: F401
from hmm import HMM_EM

LOG_2PI = math.log(2 * math.pi)


class HMMInference(HMM_EM):
    """
    This model is only used for inference, the prior / generative model are *not trained*
    """

    def __init__(self, z_dim, x_dim, nhid, temp, temp_prior, params=None, *args, **kwargs):
        super(HMMInference, self).__init__(z_dim, x_dim, *args, **kwargs)

        # encoder side
        self.inp_embedding = nn.Embedding(x_dim, nhid)
        self.encoder = torch.nn.LSTM(nhid, nhid, 1, dropout=0, bidirectional=True)
        self.enc = nn.ModuleList([self.inp_embedding, self.encoder])

        # latent stuff
        self.z_decoder = nn.GRUCell(2 * nhid, z_dim)  # This is the autoregressive q(z)
        self.logits = nn.Linear(z_dim, z_dim)
        self.temp = Variable(torch.Tensor([temp]))
        self.temp_prior = Variable(torch.Tensor([temp_prior]))

        # no more decoder - that's in the parent HMM

        self.init_weights()
        if 'params' not in kwargs or kwargs['params'] is None:
            self.randomly_initialize()
        # otherwise, it's been handled by the parent

        self.nhid = nhid

        # now we upgrade the generation parameters to be nn.Parameters
        # that don't require a gradient
        self.T = nn.Parameter(self.T.data, requires_grad=False)
        self.pi = nn.Parameter(self.pi.data, requires_grad=False)
        self.emit = nn.Parameter(self.emit.data, requires_grad=False)

    def randomly_initialize(self):
        T = np.random.random(size=(self.z_dim, self.z_dim))
        T = T/T.sum(axis=1).reshape((self.z_dim, 1))

        pi = np.random.random(size=(self.z_dim,))
        pi = pi/pi.sum()

        emit = np.random.random(size=(self.z_dim, self.x_dim))
        emit = emit/emit.sum(axis=1).reshape((self.z_dim, 1))

        self.T = nn.Parameter(torch.from_numpy(T.T).float().log())
        self.pi = nn.Parameter(torch.from_numpy(pi).float().log())
        self.emit = nn.Parameter(torch.from_numpy(emit.T).float().log())

    def cuda(self, *args, **kwargs):
        self.temp = self.temp.cuda()
        self.temp_prior = self.temp_prior.cuda()
        super(HMMInference, self).cuda(*args, **kwargs)

    def init_weights(self):
        initrange = 0.1
        self.inp_embedding.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz, dim, directions=1, squeeze=False):
        weight = next(self.parameters()).data
        if squeeze:
            return Variable(weight.new(directions, bsz, dim).zero_().squeeze())
        else:
            return (Variable(weight.new(directions, bsz, dim).zero_()),
                    Variable(weight.new(directions, bsz, dim).zero_()))

    def decode(self, z, x, emit):
        """
        Computes \log p(x | z); emit is [x_dim x z_dim], z is [batch_sz x z_dim]
        result is [batch_sz x x_dim]
        """
        probs = torch.matmul(emit, z.unsqueeze(2)).squeeze()
        return Categorical(probs=probs).log_prob(x)

    def forward(self, input, args, n_particles, test=False):
        T = nn.Softmax(dim=0)(self.T)
        pi = nn.Softmax(dim=0)(self.pi)
        emit = nn.Softmax(dim=0)(self.emit)

        # run the input and teacher-forcing inputs through the embedding layers here
        seq_len, batch_sz = input.size()
        emb = self.inp_embedding(input)
        hidden = self.init_hidden(batch_sz, self.nhid, 2)  # bidirectional
        hidden_states, (h, c) = self.encoder(emb, hidden)
        hidden_states = hidden_states.repeat(1, n_particles, 1)

        # run the z-decoder at this point, evaluating the NLL at each step
        h = self.init_hidden(batch_sz * n_particles, self.z_dim, squeeze=True)

        nlls = hidden_states.data.new(seq_len, batch_sz * n_particles)
        loss = 0

        accumulated_weights = -math.log(n_particles)  # will contain log w_{t - 1}
        resamples = 0

        prior_probs = pi.unsqueeze(0).expand(batch_sz * n_particles, self.z_dim)

        for i in range(seq_len):
            h = self.z_decoder(hidden_states[i], h)
            logits = self.logits(h)

            # build the next z sample
            if test:
                q = OneHotCategorical(logits=logits)
                z = q.sample()
            else:
                q = RelaxedOneHotCategorical(temperature=self.temp, logits=logits)
                z = q.rsample()
            h = z

            # prior
            if test:
                p = OneHotCategorical(prior_probs)
            else:
                p = RelaxedOneHotCategorical(temperature=self.temp_prior, probs=prior_probs)

            # now, compute the log-likelihood of the data given this z-sample
            NLL = -self.decode(z, input[i].repeat(n_particles), emit)  # diff. w.r.t. z
            nlls[i] = NLL.data

            # compute the weight using `reweight` on page (4)
            f_term = p.log_prob(z)  # prior
            r_term = q.log_prob(z)  # proposal
            alpha = -NLL + (f_term - r_term)

            wa = accumulated_weights + alpha.view(n_particles, batch_sz)

            # sample ancestors, and reindex everything
            Z = log_sum_exp(wa, dim=0)  # line 7
            # if (Z.data > 0.1).any():
            #     pdb.set_trace()

            loss += Z  # line 8
            accumulated_weights = wa - Z  # line 9

            if args.filter:
                probs = accumulated_weights.data.exp()
                probs += 0.01
                probs = probs / probs.sum(0, keepdim=True)
                effective_sample_size = 1./probs.pow(2).sum(0)

                # resample / RSAMP
                if ((effective_sample_size / n_particles) < 0.3).sum() > 0:
                    resamples += 1
                    ancestors = torch.multinomial(probs.transpose(0, 1), n_particles, True)

                    # now, reindex, which is the most important thing
                    offsets = n_particles * torch.arange(batch_sz).unsqueeze(1).repeat(1, n_particles).long()
                    if ancestors.is_cuda:
                        offsets = offsets.cuda()
                    unrolled_idx = Variable(ancestors.t().contiguous()+offsets).view(-1)
                    h = torch.index_select(h, 0, unrolled_idx)

                    # reset accumulated_weights
                    accumulated_weights = -math.log(n_particles)  # will contain log w_{t - 1}

            if i != seq_len - 1:
                prior_probs = torch.matmul(T, z.unsqueeze(2)).squeeze()

        # now, we calculate the final log-marginal estimator
        nll = nlls.view(seq_len, n_particles, batch_sz).mean(1).sum()
        return -loss.sum(), nll, (seq_len * batch_sz), resamples

    def evaluate(self, data_source, args, num_importance_samples=3):
        self.eval()
        old_anneal = args.anneal  # save to replace after evaluation
        args.anneal = 1.
        total_loss = 0
        total_resamples = 0
        batch_idx = 0

        true_marginal = 0

        for batch in data_source:
            if args.cuda:
                batch = batch.cuda()
            data = Variable(batch.squeeze().t().contiguous())  # squeeze for 1 billion
            elbo, _, _, resamples = self.forward(data, args, num_importance_samples, test=True)
            total_loss += elbo.detach().data
            total_resamples += resamples
            batch_idx += 1
            # true_marginal += self.log_marginal(batch.t()).sum()
        args.anneal = old_anneal

        return total_loss[0], true_marginal

    def train_epoch(self, train_data, optimizer, epoch, args, num_importance_samples):
        self.train()
        dataset_size = len(train_data)

        def train_loop(profile=False):
            total_loss = 0
            total_tokens = 0
            total_resamples = 0
            batch_idx = 0

            # for pretty printing the loss in each chunk
            last_chunk_loss = 0
            last_chunk_tokens = 0
            last_chunk_resamples = 0

            for batch in train_data:
                if args.cuda:
                    batch = batch.cuda()
                data = Variable(batch.squeeze().t().contiguous())  # squeeze for 1 billion
                if profile and batch_idx > 10:
                    print("breaking because profiling finished;")
                    break
                if epoch > args.kl_anneal_delay:
                    args.anneal = min(args.anneal + args.kl_anneal_rate, 1.)
                optimizer.zero_grad()
                elbo, NLL, tokens, resamples = self.forward(data, args, num_importance_samples)
                loss = elbo/tokens
                loss.backward()
                # torch.nn.utils.clip_grad_norm(self.parameters(), args.clip)
                optimizer.step()

                total_loss += elbo.detach()
                total_tokens += tokens
                total_resamples += resamples

                # print if necessary
                if batch_idx % args.log_interval == 0 and batch_idx > 0 and not args.quiet:
                    chunk_loss = total_loss.data[0] - last_chunk_loss
                    chunk_tokens = total_tokens - last_chunk_tokens
                    chunk_resamples = (total_resamples - last_chunk_resamples) / args.log_interval
                    print_in_epoch_summary(epoch, batch_idx, args.batch_size, dataset_size,
                                           loss.data[0], NLL / tokens,
                                           {'Chunk Loss': chunk_loss / chunk_tokens, 'resamples': chunk_resamples},
                                           tokens, "anneal={:.2f}".format(args.anneal))
                    last_chunk_loss = total_loss.data[0]
                    last_chunk_tokens = total_tokens
                    last_chunk_resamples = total_resamples
                batch_idx += 1  # because no cheap generator smh
            return total_loss.data[0], total_tokens

        if args.prof is None:
            total_loss, total_tokens = train_loop(False)
            return total_loss / total_tokens
        else:
            with torch.autograd.profiler.profile() as prof:
                _, _ = train_loop(True)
            prof.export_chrome_trace(args.prof)
            sys.exit(0)


class HMM_VI(HMMInference):
    """
    This model fits the prior, generative model, and inference jointly
    """

    def __init__(self, *args, **kwargs):
        super(HMM_VI, self).__init__(*args, **kwargs)

        # turn on gradients
        self.T = nn.Parameter(self.T.data, requires_grad=True)
        self.pi = nn.Parameter(self.pi.data, requires_grad=True)
        self.emit = nn.Parameter(self.emit.data, requires_grad=True)

class HMM_VI_Layers(HMM_VI):
    def __init__(self, z_dim, x_dim, hidden_size, nhid, temp, temp_prior, params=None, *args, **kwargs):
        super(HMM_VI, self).__init__(z_dim, x_dim, nhid, temp, temp_prior, params, *args, **kwargs)

        self.emit = nn.Parameter(torch.zeros(x_dim, hidden_size))
        self.hidden = nn.Parameter(torch.zeros(hidden_size, z_dim))

        # fix the random initialization
        self.emit.data.uniform_(-0.01, 0.01)
        self.hidden.data.uniform_(-0.01, 0.01)

    def decode(self, z, x, emit):
        """
        Computes \log p(x | z); emit is [x_dim x z_dim], z is [batch_sz x z_dim]
        result is [batch_sz x x_dim]
        """
        probs = torch.matmul(torch.mm(self.emit, self.hidden), z.unsqueeze(2)).squeeze()
        return Categorical(probs=probs).log_prob(x)
