"""
This is just broken out of hmm_filter.py because I'm sick of looking at that file
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.distributions import RelaxedOneHotCategorical, OneHotCategorical
import numpy as np
import math
import sys
import pdb  # noqa: F401
import gc
from collections import defaultdict

from src.utils import print_in_epoch_summary, log_sum_exp, any_nans, VERSION  # noqa: F401
from src.locked_dropout import LockedDropout  # noqa: F401
from src.embed_regularize import embedded_dropout  # noqa: F401
from src.hmm import HMM_EM
from src.utils import show_memusage  # noqa: F401
from src.hmm_filter import HMM_MFVI_Yoon_Deep

LOG_2PI = math.log(2 * math.pi)


def check_allocation():
    for obj in gc.get_objects():
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            if obj.size()[0] == 20000:
                return True
    return False

class HMM_Gradients(HMM_MFVI_Yoon_Deep):
    """
    This model is the main experimentation method: it follows the trajectory of the MFVI exact, but 
    for each batch it computes:
        1. gradients w.r.t. the log-marginal
        2. gradients w.r.t. the exact ELBO
        3. gradients w.r.t. sampling-based ELBO 
        4. gradients w.r.t. IWAE?
        5. gradients w.r.t. FIVO
    """
    def __init__(self, z_dim, x_dim, hidden_size, nhid, word_dim, temp, temp_prior, params=None, *args, **kwargs):
        super(HMM_Gradients, self).__init__(self, z_dim, x_dim, hidden_size, nhid, word_dim, temp, temp_prior, params=None, *args, **kwargs)
        self.gradients = {}

    def train_epoch(self, train_data, optimizer, epoch, args, num_importance_samples):
        self.train()
        for i, batch in enumerate(train_data):
            if args.cuda:
                batch = batch.cuda()
            data = Variable(batch.squeeze(0).t().contiguous())  # squeeze for 1 billion
            optimizer.zero_grad()
            self.forward(data, args, num_importance_samples, False, optimizer, i, epoch)
        return 0, 0

    def forward(self, input, args, n_particles, test, optimizer=None, i=0, epoch=0):
        # precompute the encoder outputs, so we only have to do this once
        emb = self.inp_embedding(input)
        hidden = self.init_hidden(batch_sz, self.nhid, 2)  # bidirectional
        hidden_states, (_, _) = self.encoder(emb, hidden)

        if test:
            return self.exact_elbo(input, args, n_particles, test, emb, hidden_states)

        if i % 10 == 0:
            self.exact_marginal(input, args)
            self.collect_gradients(i, 'exact_marginal', optimizer, clean=True)

            # in a loop, collect both the sampled ELBO and the sampled PF gradients
            for i in range(10):
                loss = self.sampled_elbo(input, args, n_particles)
                self.collect_gradients(i, 'sampled_elbo', optimizer, clean=True)
                self.sampled_iwae(input, args, n_particles, loss)
                self.collect_gradients(i, 'sampled_iwae', optimizer, clean=True)

                # particle filter goes here

        # now do the MFVI gradient
        loss, _, tokens, _ = self.exact_elbo(input, args, n_particles, emb, hidden_states)
        loss.backward()

        if i % 10 == 0:
            self.collect_gradients(i, 'exact_elbo', optimizer, clean=False)
            self.save_gradients(epoch, i, args.base_filename)
        optimizer.step()

    def collect_gradients(self, method, optimizer, clean=False):
        if method not in self.gradients.keys():
            self.gradients[method] = defaultdict(list)
        for name, param in self.named_parameters():
            self.gradients[method][name].append(param.grad.data.cpu())
        if clean:
            optimizer.zero_grad()

    def save_gradients(self, epoch, batch_idx, filename):
        output = defaultdict(dict)
        for method in self.gradients.keys():
            for key, values in self.gradients[method].items():
                n = len(values)
                mean = sum(values)/n
                var = sum([(x - mean).pow(2) for x in values])/n
                output[method][key] = (mean, var)
        mode = 'w' if VERSION[0] == 2 else 'wb'
        path = "{}_{}_{}.pt".format(filename, epoch, batch_idx)
        with open(path, mode) as f:
            torch.save(output, f)
            print("saved gradients to " + path)

    # below this is inference methods / techniques

    #
    # EXACT ELBO
    #

    def exact_elbo(self, input, args, test, emb, hidden_states):
        seq_len, batch_sz = input.size()
        T = F.log_softmax(self.T, 0)  # NOTE: in log-space
        pi = F.log_softmax(self.pi, 0)  # in log-space, intentionally
        emit = self.calc_emit()  # also in log-space

        elbo = 0
        NLL = 0

        # now a logit
        prior_logits = pi.unsqueeze(0).expand(batch_sz, self.z_dim)

        prev_probs = None

        for i in range(seq_len):
            logits = F.log_softmax(self.logits(hidden_states[i]), 1)  # log q(z_i)
            probs = logits.exp()  # q(z_i)
            emission = F.embedding(input[i]), emit)  # log p(x_i | z_i)

            # unary potentials
            elbo += (emission * probs).sum(1)  # E_q[log p(x_i | z_i)]
            NLL += -(emission * probs).sum(1).data

            # binary potentials q(z_t)q(z_{t - 1})log p(z_t | z_{t - 1})
            if i != 0:
                elbo += (prev_probs.unsqueeze(1) *
                         probs.unsqueeze(2) *
                         T.unsqueeze(0)).sum(2).sum(1)
            else:
                # add the log p(z_1) term
                elbo += (probs * prior_logits).sum(1)

            # entropy term - E[-log q]
            elbo -= (logits * probs).sum(1)

            prev_probs = probs

        # now, we calculate the final log-marginal estimator
        return -elbo.sum(), NLL.sum(), (seq_len * batch_sz), 0

    #
    # EXACT MARGINAL
    #

    def exact_marginal(self, input):
        seq_len, batch_sz = input.size()
        _, _, log_marginal = self.forward_backward(input, speedup=True)
        loss = -log_marginal.sum() / (seq_len * batch_sz)
        loss.backward(retain_graph=True)

    def forward_backward(self, input, speedup=False):
        """
        Modify the forward-backward to compute beta[t], since we need that for checking the sampling in the particle filter case
        """
        input = input.long()

        seq_len, batch_size = input.size()
        alpha = [None for i in range(seq_len)]
        beta = [None for i in range(seq_len)]

        T = F.log_softmax(self.T, 0)
        pi = F.log_softmax(self.pi, 0)
        emit = self.calc_emit()

        # forward pass
        alpha[0] = self.log_prob(input[0], (emit,)) + pi.view(1, -1)
        beta[-1] = Variable(torch.zeros(batch_size, self.z_dim))

        if T.is_cuda:
            beta[-1] = beta[-1].cuda()

        for t in range(1, seq_len):
            logprod = alpha[t - 1].unsqueeze(2).expand(batch_size, self.z_dim, self.z_dim) + T.t().unsqueeze(0)
            alpha[t] = self.log_prob(input[t], (emit,)) + log_sum_exp(logprod, 1)

        log_marginal = log_sum_exp(alpha[-1] + beta[-1], dim=-1)

        if speedup:
            return 0, 0, log_marginal
        else:
            for t in range(seq_len - 2, -1, -1):
                beta[t] = log_sum_exp(T.unsqueeze(0) +
                                      beta[t + 1].unsqueeze(2) +
                                      F.embedding(input[t + 1], emit).unsqueeze(2), 1)

            return [alpha[i] + beta[i] - log_marginal.unsqueeze(1) for i in range(seq_len)], 0, log_marginal

    # 
    # SAMPLED ELBO 
    #

    def sampled_elbo(input, args, n_particles, emb, hidden_states):
        seq_len, batch_sz = input.size()
        T = nn.Softmax(dim=0)(self.T)  # NOTE: not in log-space
        pi = nn.Softmax(dim=0)(self.pi)
        emit = self.calc_emit()

        hidden_states = hidden_states.repeat(1, n_particles, 1)
        nlls = hidden_states.data.new(seq_len, batch_sz * n_particles)
        loss = 0

        # now a value in probability space
        prior_probs = pi.unsqueeze(0).expand(batch_sz * n_particles, self.z_dim)

        logits = self.init_hidden(batch_sz * n_particles, self.z_dim, squeeze=True)

        for i in range(seq_len):
            logits = self.logits(hidden_states[i])

            # build the next z sample
            # p = RelaxedOneHotCategorical(probs=prior_probs, temperature=Variable(torch.Tensor([args.temp_prior]).cuda()))
            q = RelaxedOneHotCategorical(temperature=Variable(torch.Tensor([args.temp_prior]).cuda()), logits=logits)
            z = q.sample()

            log_probs = F.log_softmax(logits, dim=1)

            # now, compute the log-likelihood of the data given this z-sample
            # so emission is [batch_sz x z_dim], i.e. emission[i, j] is the log-probability of getting this
            # data for element i given choice z
            emission = F.embedding(input[i].repeat(n_particles), emit)

            NLL = -log_sum_exp(emission + log_probs, 1)
            nlls[i] = NLL.data
            KL = (log_probs.exp() * (log_probs - (prior_probs + 1e-16).log())).sum(1)
            loss += (NLL + KL)

            if i != seq_len - 1:
                prior_probs = (T.unsqueeze(0) * z.unsqueeze(1)).sum(2)

        (loss.sum()/(seq_len * batch_sz * n_particles)).backward(retain_graph=True)
        return loss

    #
    # SAMPLED IWAE
    #

    def sampled_iwae(input, args, n_particles, loss):
        loss = log_sum_exp(-loss.view(n_particles, batch_sz), 0) + math.log(n_particles)
        loss.sum().backward(retain_graph=True)

