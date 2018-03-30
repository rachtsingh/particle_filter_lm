# This is just broken out of hmm_filter.py because I'm sick of looking at that file

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.distributions import RelaxedOneHotCategorical, OneHotCategorical
import numpy as np
import math
import pdb  # noqa: F401
import gc
from collections import defaultdict

from src.utils import print_in_epoch_summary, log_sum_exp, any_nans, VERSION  # noqa: F401
from src.locked_dropout import LockedDropout  # noqa: F401
from src.embed_regularize import embedded_dropout  # noqa: F401
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
    def __init__(self, *args, **kwargs):
        super(HMM_Gradients, self).__init__(*args, **kwargs)
        self.gradients = {}
        self.storage = {}

    def train_epoch(self, train_data, optimizer, epoch, args, num_importance_samples):
        self.train()
        for i, batch in enumerate(train_data):
            if (i + 1) % 250 == 0:
                print(i + 1)
            if args.cuda:
                batch = batch.cuda()
            data = Variable(batch.squeeze(0).t().contiguous())  # squeeze for 1 billion
            optimizer.zero_grad()
            self.forward(data, args, num_importance_samples, False, optimizer, i, epoch)
        return 0, 0

    def forward(self, input, args, n_particles, test, optimizer=None, i=0, epoch=0):
        # avoid precomputation because it's expensive
        if args.train_method == 'exact_marginal':
            return self.switch_methods(input, args, n_particles, None, None, optimizer)

        # precompute the encoder outputs, so we only have to do this once
        seq_len, batch_sz = input.size()
        emb = self.inp_embedding(input)
        hidden = self.init_hidden(batch_sz, self.nhid, 2)  # bidirectional
        hidden_states, (_, _) = self.encoder(emb, hidden)

        if test:
            # this is super ugly
            # if args.train_method == 'sampled_filter':
                # return self.sampled_filter(input, args, n_particles, emb, hidden_states)
            return self.exact_elbo(input, args, test, emb, hidden_states)

        if args.train_method:
            # not trying to get gradients, just try to dump the trajectory
            return self.switch_methods(input, args, n_particles, emb, hidden_states, optimizer)

        if i % 10 == 0:
            self.exact_marginal(input)
            self.collect_gradients('exact_marginal', optimizer, clean=True)

            # in a loop, collect both the sampled ELBO and the sampled PF gradients
            for k in range(20):
                loss, _, tokens, _ = self.sampled_elbo(input, args, n_particles, emb, hidden_states)
                self.collect_gradients('sampled_elbo', optimizer, clean=True)
                self.sampled_iwae(input, args, n_particles, loss, tokens)
                self.collect_gradients('sampled_iwae', optimizer, clean=True)
                self.sampled_filter(input, args, n_particles, emb, hidden_states)
                self.collect_gradients('sampled_filter', optimizer, clean=True)

        # now do the MFVI gradient
        loss, _, tokens, _ = self.exact_elbo(input, args, test, emb, hidden_states)
        (loss/tokens).backward()

        if i % 10 == 0:
            self.collect_gradients('exact_elbo', optimizer, clean=False)
            self.save_gradients(epoch, i, args.base_filename)
        optimizer.step()

    def switch_methods(self, input, args, n_particles, emb, hidden_states, optimizer):
        if args.train_method == 'exact_elbo':
            loss, _, tokens, _ = self.exact_elbo(input, args, False, emb, hidden_states)
            (loss/tokens).backward()
        elif args.train_method == 'exact_marginal':
            self.exact_marginal(input)
        elif args.train_method == 'sampled_elbo':
            self.sampled_elbo(input, args, n_particles, emb, hidden_states)
        elif args.train_method == 'sampled_iwae':
            loss, _, tokens, _ = self.sampled_elbo(input, args, n_particles, emb, hidden_states)
            optimizer.zero_grad()
            self.sampled_iwae(input, args, n_particles, loss, tokens)
        elif args.train_method == 'sampled_filter':
            self.sampled_filter(input, args, n_particles, emb, hidden_states)
        optimizer.step()

    def collect_gradients(self, method, optimizer, clean=False):
        if method not in self.gradients.keys():
            self.gradients[method] = defaultdict(list)
        for name, param in self.named_parameters():
            if not(param.grad is None):  # for the exact marginal estimator, there's no grad for the inference net
                self.gradients[method][name].append(param.grad.data.clone())
        if clean:
            optimizer.zero_grad()

    def get_parameter_statistics(self):
        gen_parameters = [v for k, v in self.named_parameters() if k in ('T', 'pi', 'emit', 'hidden')]
        inf_parameters = [v for k, v in self.named_parameters() if k not in ('T', 'pi', 'emit', 'hidden')]
        gen_count = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in gen_parameters)
        inf_count = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in inf_parameters)
        return (gen_count, inf_count)

    def save_gradients(self, epoch, batch_idx, filename):
        summary = defaultdict(dict)
        gen_parameters, inf_parameters = self.get_parameter_statistics()
        for name, _ in self.named_parameters():
            # first we get the values out of exact_elbo and exact_marginal to understand
            if name in ('T', 'pi', 'emit', 'hidden'):
                summary['exact_marginal'][name] = (self.gradients['exact_marginal'][name][0], None)
            summary['exact_elbo'][name] = (self.gradients['exact_elbo'][name][0], None)

            # now the others
            for method in self.gradients.keys():
                if method in ('exact_elbo', 'exact_marginal'):
                    pass
                for key, values in self.gradients[method].items():
                    n = len(values)
                    mean = sum(values)/n
                    var = (sum([(x - mean).pow(2) for x in values])/n).sqrt()
                    summary[method][key] = (mean, var)  # these are still high dimensional CUDA vectors

        # now compute the statistics to save
        output = {}

        for method in self.gradients.keys():
            gen_bias = 0
            gen_bias_elbo = 0  # based on the exact elbo
            inf_bias = 0
            gen_var = 0
            inf_var = 0
            gen_snr = 0
            gen_elbo_snr = 0
            inf_snr = 0

            gen_counted = 0.01
            inf_counted = 0.01

            for name, _ in self.named_parameters():
                if method == 'exact_marginal':
                    continue
                mean, var = summary[method][name]

                # we have to mask out values for which the variance is 0
                msk = (var.abs() > 1e-12)
                if name in ('T', 'pi', 'emit', 'hidden'):
                    gen_bias += (mean - summary['exact_marginal'][name][0]).pow(2).sum()
                    gen_bias_elbo += (mean - summary['exact_elbo'][name][0]).pow(2).sum()
                    gen_var += var.sum()
                    gen_snr += ((mean[msk] - summary['exact_marginal'][name][0][msk]).abs() / var[msk]).sum()
                    gen_elbo_snr += ((mean[msk] - summary['exact_elbo'][name][0][msk]).abs() / var[msk]).sum()
                    gen_counted += msk.sum()
                else:
                    inf_bias += (mean - summary['exact_elbo'][name][0]).pow(2).sum()
                    inf_var += var.sum()
                    inf_snr += ((mean[msk] - summary['exact_elbo'][name][0][msk]).abs() / var[msk]).sum()
                    inf_counted += msk.sum()

            # scale down by the norm of the difference
            gen_bias = np.sqrt(gen_bias / gen_parameters)  # L2 norm of difference
            inf_bias = np.sqrt(inf_bias / inf_parameters)
            gen_bias_elbo = np.sqrt(gen_bias_elbo / gen_parameters)
            gen_var = (gen_var / gen_parameters)  # still squared L2 norm of difference
            inf_var = (inf_var / inf_parameters)
            gen_snr /= (gen_counted)
            gen_elbo_snr /= (gen_counted)
            inf_snr /= (inf_counted)

            output[method] = (gen_bias, gen_bias_elbo, inf_bias, gen_var, inf_var, gen_snr, gen_elbo_snr, inf_snr)

        # save everything
        mode = 'w' if VERSION[0] == 2 else 'wb'
        path = "{}_{}_{}.pt".format(filename, epoch, batch_idx)
        with open(path, mode) as f:
            torch.save(output, f)
            # print("saved gradients to " + path)
        self.gradients = {}
        self.storage[path] = output

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
            emission = F.embedding(input[i], emit)  # log p(x_i | z_i)

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

    def eval_exact_marginal(self, input):
        seq_len, batch_sz = input.size()
        _, _, log_marginal = self.forward_backward(input, speedup=True)
        return -log_marginal.sum(), 0, (seq_len * batch_sz), 0

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
    def sampled_elbo(self, input, args, n_particles, emb, hidden_states):
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
            p = RelaxedOneHotCategorical(temperature=self.temp_prior, probs=prior_probs)
            q = RelaxedOneHotCategorical(temperature=self.temp, logits=logits)
            z = q.rsample()

            log_probs = F.log_softmax(logits, dim=1)

            # now, compute the log-likelihood of the data given this z-sample
            # so emission is [batch_sz x z_dim], i.e. emission[i, j] is the log-probability of getting this
            # data for element i given choice z
            emission = F.embedding(input[i].repeat(n_particles), emit)

            NLL = -log_sum_exp(emission + log_probs, 1)
            nlls[i] = NLL.data
            KL = q.log_prob(z) - p.log_prob(z)  # pretty inexact
            loss += (NLL + KL)

            if i != seq_len - 1:
                prior_probs = (T.unsqueeze(0) * z.unsqueeze(1)).sum(2)

        (loss.sum()/(seq_len * batch_sz * n_particles)).backward(retain_graph=True)
        return loss, 0, seq_len * batch_sz * n_particles, 0

    #
    # SAMPLED IWAE
    #
    def sampled_iwae(self, input, args, n_particles, loss, tokens):
        seq_len, batch_sz = input.size()
        loss = -log_sum_exp(-loss.view(n_particles, batch_sz), 0) + math.log(n_particles)
        (loss.sum()/tokens).backward(retain_graph=True)

    def evaluate(self, data_source, args, num_importance_samples=3):
        if args.train_method == 'exact_marginal':
            return self.exact_evaluate(data_source, args, num_importance_samples)
        else:
            return super(HMM_Gradients, self).evaluate(data_source, args, num_importance_samples)

    #
    # SAMPLED PARTICLE FILTER METHOD
    #
    def sampled_filter(self, input, args, n_particles, emb, hidden_states):
        seq_len, batch_sz = input.size()
        T = F.log_softmax(self.T, 0)  # NOTE: in log-space
        pi = F.log_softmax(self.pi, 0)  # NOTE: in log-space
        emit = self.calc_emit()

        hidden_states = hidden_states.repeat(1, n_particles, 1)

        nlls = hidden_states.data.new(seq_len, batch_sz * n_particles)
        loss = 0

        accumulated_weights = -math.log(n_particles)  # will contain log w_{t - 1}
        resamples = 0

        # in log probability space
        prior_logits = pi.unsqueeze(0).expand(batch_sz * n_particles, self.z_dim)

        for i in range(seq_len):
            # the approximate posterior comes from the same thing as before
            logits = self.logits(hidden_states[i])

            if not self.training:
                # this is crucial!!
                p = OneHotCategorical(logits=prior_logits)
                q = OneHotCategorical(logits=logits)
                z = q.sample()
            else:
                p = RelaxedOneHotCategorical(temperature=self.temp_prior, logits=prior_logits)
                q = RelaxedOneHotCategorical(temperature=self.temp, logits=logits)
                z = q.rsample()

            # now, compute the log-likelihood of the data given this z-sample
            emission = F.embedding(input[i].repeat(n_particles), emit)
            NLL = -(emission * z).sum(1)
            # NLL = -self.decode(z, input[i].repeat(n_particles), (emit,))  # diff. w.r.t. z
            nlls[i] = NLL.data

            # compute the weight using `reweight` on page (4)
            f_term = p.log_prob(z)  # prior
            r_term = q.log_prob(z)  # proposal
            alpha = -NLL + (f_term - r_term)

            wa = accumulated_weights + alpha.view(n_particles, batch_sz)

            Z = log_sum_exp(wa, dim=0)  # line 7

            loss += Z  # line 8
            accumulated_weights = wa - Z  # F.log_softmax(wa, dim=0)  # line 9

            # sample ancestors, and reindex everything
            if args.filter:
                probs = accumulated_weights.data.exp()
                probs += 0.01
                probs = probs / probs.sum(0, keepdim=True)
                effective_sample_size = 1./probs.pow(2).sum(0)

                # probs is [n_particles, batch_sz]
                # ancestors [2 x 15] = [[0, 0, 0, ..., 0], [0, 1, 2, 3, ...]]
                # offsets   [2 x 15] = [[0, 0, 0, ..., 0], [1, 1, 1, 1, ...]]

                # resample / RSAMP
                if ((effective_sample_size / n_particles) < 0.3).sum() > 0:
                    resamples += 1
                    ancestors = torch.multinomial(probs.transpose(0, 1), n_particles, True)

                    # now, reindex, which is the most important thing
                    offsets = n_particles * torch.arange(batch_sz).unsqueeze(1).repeat(1, n_particles).long()
                    if ancestors.is_cuda:
                        offsets = offsets.cuda()
                    unrolled_idx = Variable(ancestors + offsets).view(-1)
                    z = torch.index_select(z, 0, unrolled_idx)

                    # reset accumulated_weights
                    accumulated_weights = -math.log(n_particles)  # will contain log w_{t - 1}

            if i != seq_len - 1:
                # now in log-probability space
                prior_logits = log_sum_exp(T.unsqueeze(0) + z.unsqueeze(1), 2)

        if self.training:
            (-loss.sum()/(seq_len * batch_sz * n_particles)).backward(retain_graph=True)
        return -loss.sum(), nlls.sum(), seq_len * batch_sz * n_particles, 0

    # override because otherwise it's slow
    def exact_evaluate(self, data_source, args, num_importance_samples=3):
        self.eval()
        total_log_marginal = 0
        total_tokens = 0

        for batch in data_source:
            if args.cuda:
                batch = batch.cuda()
            batch = batch.squeeze(0).t().contiguous()
            data = Variable(batch)  # squeeze for 1 billion
            total_log_marginal += self.eval_log_marginal(batch).sum()
            total_tokens += (data.size()[0] * data.size()[1])

        if args.dataset != '1billion':
            total_tokens = 1

        return total_log_marginal / total_tokens, 0, total_log_marginal / total_tokens
