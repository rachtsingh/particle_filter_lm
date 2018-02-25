"""
This is an attempt to do inference on HMMs using particle filtering VI
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import RelaxedOneHotCategorical, OneHotCategorical, Categorical
import math
import sys
import pdb  # noqa: F401

from utils import print_in_epoch_summary, log_sum_exp, any_nans  # noqa: F401
from locked_dropout import LockedDropout  # noqa: F401
from embed_regularize import embedded_dropout  # noqa: F401
from hmm import HMM

LOG_2PI = math.log(2 * math.pi)


class HMMInference(HMM):
    """
    This model is only used for inference, the prior / generative model are *not trained*
    """

    def __init__(self, z_dim, x_dim, nhid, *args, **kwargs):
        super(HMMInference, self).__init__(z_dim, x_dim, *args, **kwargs)

        # encoder side
        self.inp_embedding = nn.Embedding(x_dim, nhid)
        self.encoder = torch.nn.LSTM(nhid, nhid, 1, dropout=0, bidirectional=True)
        self.enc = nn.ModuleList([self.inp_embedding, self.encoder])

        # latent stuff
        self.z_decoder = nn.GRUCell(2 * nhid, z_dim)  # This is the autoregressive q(z)
        self.logits = nn.Linear(z_dim, z_dim)
        self.temp = Variable(torch.Tensor([0.67]))
        self.temp_prior = Variable(torch.Tensor([0.5]))

        # no more decoder - that's in the parent HMM

        self.init_weights()
        self.nhid = nhid

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

    def decode(self, z, x):
        """
        Computes \log p(x | z); emit is [x_dim x z_dim], z is [batch_sz x z_dim]
        result is [batch_sz x x_dim]
        """
        probs = torch.matmul(self.emit, z.data.unsqueeze(2)).squeeze()
        return Categorical(probs=Variable(probs)).log_prob(x)

    def forward(self, input, targets, args, n_particles, test=False):
        """
        This version takes the inputs, and does not expose the logits, but instead
        computes the losses directly
        """

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

        prior_probs = Variable(self.pi.unsqueeze(0).expand(batch_sz * n_particles, self.z_dim))

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
            p = OneHotCategorical(prior_probs)

            # now, compute the log-likelihood of the data given this z-sample
            NLL = self.decode(z, input[i].repeat(n_particles))  # diff. w.r.t. z
            nlls[i] = NLL.data

            # compute the weight using `reweight` on page (4)
            f_term = p.log_prob(z)  # prior
            r_term = q.log_prob(z)  # proposal
            alpha = -NLL + args.anneal * (f_term - r_term)

            wa = accumulated_weights + alpha.view(n_particles, batch_sz)

            # sample ancestors, and reindex everything
            Z = log_sum_exp(wa, dim=0)  # line 7
            if (Z.data > 0.1).any():
                pdb.set_trace()

            loss += Z  # line 8
            accumulated_weights = wa - Z  # line 9
            probs = accumulated_weights.data.exp()
            probs += 0.01
            probs = probs / probs.sum(0, keepdim=True)
            effective_sample_size = 1./probs.pow(2).sum(0)

            # resample / RSAMP if 3 batch elements need resampling
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
                prior_probs = Variable(torch.matmul(self.T, z.data.unsqueeze(2)).squeeze())

        # now, we calculate the final log-marginal estimator
        nll = nlls.view(seq_len, n_particles, batch_sz).mean(1).sum()
        return -loss.sum(), nll, (seq_len * batch_sz), resamples

    def evaluate(self, data_source, args, num_importance_samples=3):
        self.eval()
        old_anneal = args.anneal  # save to replace after evaluation
        args.anneal = 1.
        old_temps = self.temp, self.temp_prior
        self.temp = Variable(self.temp.data.new([0]))
        self.temp_prior = Variable(self.temp_prior.data.new([0]))
        total_loss = 0
        total_nll = 0
        total_tokens = 0
        total_resamples = 0
        batch_idx = 0

        for batch in data_source:
            data, targets = batch.text, batch.target
            elbo, NLL, tokens, resamples = self.forward(data, targets, args, num_importance_samples, test=True)
            total_loss += elbo.detach().data
            total_nll += NLL
            total_tokens += tokens
            total_resamples += resamples
            batch_idx += 1
        print("eval: {:.3f} NLL | current anneal: {:.3f} | average resamples: {:.1f}".format(total_nll / total_loss[0],
                                                                                             old_anneal,
                                                                                             total_resamples/batch_idx))
        args.anneal = old_anneal
        self.temp, self.temp_prior = old_temps

        # duplicate total_loss because we don't have a separate ELBO loss here, though we can grab it
        return total_loss[0] / total_tokens, total_loss[0] / total_tokens, total_nll / total_tokens

    def train_epoch(self, train_data, optimizer, epoch, args, num_importance_samples):
        self.train()
        dataset_size = len(train_data.data)

        if epoch <= args.kl_anneal_delay:
            num_importance_samples = 2   # less need for a filter if you're just pretraining the generation net

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
                if profile and batch_idx > 10:
                    print("breaking because profiling finished;")
                    break
                if epoch > args.kl_anneal_delay:
                    args.anneal = min(args.anneal + args.kl_anneal_rate, 1.)
                optimizer.zero_grad()
                data, targets = batch.text, batch.target
                elbo, NLL, tokens, resamples = self.forward(data, targets, args, num_importance_samples)
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
            return total_loss.data[0], total_tokens

        if args.prof is None:
            total_loss, total_tokens = train_loop(False)
            return total_loss / total_tokens
        else:
            with torch.autograd.profiler.profile() as prof:
                _, _ = train_loop(True)
            prof.export_chrome_trace(args.prof)
            sys.exit(0)
