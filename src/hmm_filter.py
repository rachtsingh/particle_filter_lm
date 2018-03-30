"""
This is an attempt to do inference on HMMs using particle filtering VI

Not all models do particle filtering - we don't in the marginalized model yet because it hasn't been
implemented, and the mean field model doesn't do filtering because it doesn't make sense
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

from src.utils import print_in_epoch_summary, log_sum_exp, any_nans, VERSION  # noqa: F401
from src.locked_dropout import LockedDropout  # noqa: F401
from src.embed_regularize import embedded_dropout  # noqa: F401
from src.hmm import HMM_EM
from src.utils import show_memusage  # noqa: F401

LOG_2PI = math.log(2 * math.pi)


def check_allocation():
    for obj in gc.get_objects():
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            if obj.size()[0] == 20000:
                return True
    return False


class HMMInference(HMM_EM):
    """
    This model is only used for inference, the prior / generative model are *not trained*
    """

    def __init__(self, z_dim, x_dim, nhid, word_dim, temp, temp_prior, params=None, *args, **kwargs):
        super(HMMInference, self).__init__(z_dim, x_dim, *args, **kwargs)

        # encoder side
        self.inp_embedding = nn.Embedding(x_dim, word_dim)
        self.encoder = torch.nn.LSTM(word_dim, nhid, 1, dropout=0, bidirectional=True)
        self.enc = nn.ModuleList([self.inp_embedding, self.encoder])

        # latent stuff
        self.z_decoder = nn.GRUCell(2 * nhid + z_dim, z_dim)  # This is the autoregressive q(z)
        self.logits = nn.Linear(z_dim, z_dim)

        # self.logits = nn.Linear(z_dim + 2 * nhid, z_dim)
        self.temp = Variable(torch.Tensor([temp]))
        self.temp_prior = Variable(torch.Tensor([temp_prior]))

        # no more decoder - that's in the parent HMM

        self.init_weights()
        if 'params' not in kwargs or kwargs['params'] is None:
            self.randomly_initialize()
        # otherwise, it's been handled by the parent

        self.word_dim = word_dim
        self.nhid = nhid

        # now we upgrade the generation parameters to be nn.Parameters
        # that don't require a gradient
        self.T = nn.Parameter(self.T.data, requires_grad=False)
        self.pi = nn.Parameter(self.pi.data, requires_grad=False)
        self.emit = nn.Parameter(self.emit.data, requires_grad=False)

    def load_embedding(self, embedding):
        x_dim, word_dim = embedding.size()
        if x_dim != self.x_dim or word_dim != self.word_dim:
            raise ValueError("embedding has size: {} when expected: ({}, {})".format(embedding.size(), self.x_dim, self.word_dim))
        self.inp_embedding.weight.data = embedding

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

    def decode(self, z, x, precompute):
        """
        Computes \log p(x | z); emit is [x_dim x z_dim], z is [batch_sz x z_dim]
        result is [batch_sz x x_dim]
        """
        emit, = precompute
        emission = F.embedding(x, emit)
        return (emission * z).sum(1)

    def calc_emit(self):
        return F.log_softmax(self.emit, 0)

    def forward(self, input, args, n_particles, test=False):
        T = F.log_softmax(self.T, 0)  # NOTE: in log-space
        pi = F.log_softmax(self.pi, 0)  # NOTE: in log-space
        emit = self.calc_emit()

        # run the input and teacher-forcing inputs through the embedding layers here
        seq_len, batch_sz = input.size()
        emb = self.inp_embedding(input)
        hidden = self.init_hidden(batch_sz, self.nhid, 2)  # bidirectional
        hidden_states, (_, _) = self.encoder(emb, hidden)
        hidden_states = hidden_states.repeat(1, n_particles, 1)

        # run the z-decoder at this point, evaluating the NLL at each step
        h = self.init_hidden(batch_sz * n_particles, self.z_dim, squeeze=True)

        nlls = hidden_states.data.new(seq_len, batch_sz * n_particles)
        loss = 0

        accumulated_weights = -math.log(n_particles)  # will contain log w_{t - 1}
        resamples = 0

        # in log probability space
        prior_probs = pi.unsqueeze(0).expand(batch_sz * n_particles, self.z_dim)

        logits = self.init_hidden(batch_sz * n_particles, self.z_dim, squeeze=True)

        for i in range(seq_len):
            # logits = self.logits(nn.functional.relu(self.z_decoder(hidden_states[i], logits)))
            logits = self.logits(nn.functional.relu(self.z_decoder(torch.cat([hidden_states[i], h], 1), logits)))

            # build the next z sample
            if any_nans(logits):
                pdb.set_trace()
            if test:
                q = OneHotCategorical(logits=logits)
                z = q.sample()
            else:
                q = RelaxedOneHotCategorical(temperature=self.temp, logits=logits)
                z = q.rsample()
            h = z

            # prior
            if any_nans(prior_probs):
                pdb.set_trace()
            if test:
                p = OneHotCategorical(logits=prior_probs)
            else:
                p = RelaxedOneHotCategorical(temperature=self.temp_prior, logits=prior_probs)

            if any_nans(prior_probs):
                pdb.set_trace()
            if any_nans(logits):
                pdb.set_trace()

            # now, compute the log-likelihood of the data given this z-sample
            NLL = -self.decode(z, input[i].repeat(n_particles), (emit,))  # diff. w.r.t. z
            nlls[i] = NLL.data

            # compute the weight using `reweight` on page (4)
            f_term = p.log_prob(z)  # prior
            r_term = q.log_prob(z)  # proposal
            alpha = -NLL + (f_term - r_term)

            wa = accumulated_weights + alpha.view(n_particles, batch_sz)

            # sample ancestors, and reindex everything
            Z = log_sum_exp(wa, dim=0)  # line 7

            loss += Z  # line 8
            accumulated_weights = wa - Z  # line 9

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
                    h = torch.index_select(h, 0, unrolled_idx)

                    # reset accumulated_weights
                    accumulated_weights = -math.log(n_particles)  # will contain log w_{t - 1}

            if i != seq_len - 1:
                # now in probability space
                prior_probs = log_sum_exp(T.unsqueeze(0) + z.unsqueeze(1), 2)

                # let's normalize things - slower, but safer
                # prior_probs += 0.01
                # prior_probs = prior_probs / prior_probs.sum(1, keepdim=True)

            # # if ((prior_probs.sum(1) - 1) > 1e-3).any()[0]:
            #     pdb.set_trace()

        if any_nans(loss):
            pdb.set_trace()

        # now, we calculate the final log-marginal estimator
        return -loss.sum(), nlls.sum(), (seq_len * batch_sz * n_particles), resamples

    def evaluate(self, data_source, args, num_importance_samples=3):
        self.eval()
        old_anneal = args.anneal  # save to replace after evaluation
        args.anneal = 1.
        total_loss = 0
        total_log_marginal = 0
        total_nll = 0
        total_resamples = 0
        batch_idx = 0
        total_tokens = 0

        for batch in data_source:
            if args.cuda:
                batch = batch.cuda()
            batch = batch.squeeze(0).t().contiguous()
            data = Variable(batch)  # squeeze for 1 billion
            elbo, nll, _, resamples = self.forward(data, args, num_importance_samples, test=True)
            total_log_marginal += self.eval_log_marginal(batch).sum()
            total_loss += elbo.sum().detach().data
            total_nll += nll
            total_tokens += (data.size()[0] * data.size()[1])
            total_resamples += resamples
            batch_idx += 1

        if args.dataset != '1billion':
            total_tokens = 1
        args.anneal = old_anneal

        if VERSION[1]:
            total_loss = total_loss.item()
        else:
            total_loss = total_loss[0]

        return total_loss / total_tokens, total_nll / total_tokens, total_log_marginal / total_tokens

    def train_epoch(self, train_data, optimizer, epoch, args, num_importance_samples):
        self.train()
        dataset_size = len(train_data)

        def train_loop(profile=False):
            total_loss = 0
            total_nll = 0
            total_tokens = 0
            total_resamples = 0
            batch_idx = 0

            # for pretty printing the loss in each chunk
            last_chunk_loss = 0
            last_chunk_nll = 0
            last_chunk_tokens = 0
            last_chunk_resamples = 0

            for i, batch in enumerate(train_data):
                if args.cuda:
                    batch = batch.cuda()
                data = Variable(batch.squeeze(0).t().contiguous())  # squeeze for 1 billion
                if i > 500:
                    break
                if profile and batch_idx > 10:
                    print("breaking because profiling finished;")
                    break
                if epoch > args.kl_anneal_delay:
                    args.anneal = min(args.anneal + args.kl_anneal_rate, 1.)
                optimizer.zero_grad()
                elbo, NLL, tokens, resamples = self.forward(data, args, num_importance_samples)
                loss = elbo/tokens
                loss.backward()
                # torch.nn.utils.clip_grad_norm(self.parameters(), 0.01)
                optimizer.step()

                if VERSION[1]:
                    total_loss += elbo.detach().data.item()
                else:
                    total_loss += elbo.detach().data[0]
                total_nll += NLL
                total_tokens += tokens
                total_resamples += resamples

                # print if necessary
                if batch_idx % args.log_interval == 0 and batch_idx > 0 and not args.quiet:
                    l = loss.data.item() if VERSION[1] else loss.data[0]
                    chunk_loss = total_loss - last_chunk_loss
                    chunk_nll = total_nll - last_chunk_nll
                    chunk_tokens = total_tokens - last_chunk_tokens
                    chunk_resamples = (total_resamples - last_chunk_resamples) / args.log_interval
                    print_in_epoch_summary(epoch, batch_idx, args.batch_size, dataset_size,
                                           l, NLL / tokens,
                                           {'Chunk Loss': chunk_loss / chunk_tokens,
                                            'resamples': chunk_resamples,
                                            'Chunk NLL': chunk_nll / chunk_tokens},
                                           tokens, "anneal={:.2f}".format(args.anneal))
                    last_chunk_loss = total_loss
                    last_chunk_nll = total_nll
                    last_chunk_tokens = total_tokens
                    last_chunk_resamples = total_resamples
                batch_idx += 1  # because no cheap generator smh
            return total_loss, total_tokens

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
    def __init__(self, z_dim, x_dim, hidden_size, nhid, word_dim, temp, temp_prior, params=None, *args, **kwargs):
        super(HMM_VI_Layers, self).__init__(z_dim, x_dim, nhid, word_dim, temp, temp_prior, params, *args, **kwargs)

        self.emit = nn.Parameter(torch.zeros(x_dim, hidden_size))
        self.hidden = nn.Parameter(torch.zeros(hidden_size, z_dim))

        self.hidden_size = hidden_size

        # fix the random initialization
        self.emit.data.uniform_(-0.01, 0.01)
        self.hidden.data.uniform_(-0.01, 0.01)

    def calc_emit(self):
        return F.log_softmax(torch.mm(self.emit, self.hidden), 0)

    def load_embedding(self, embedding):
        super(HMM_VI_Layers, self).load_embedding(embedding)
        self.inp_embedding.weight.data[:4] = torch.randn(4, self.word_dim)

        if self.hidden_size == self.word_dim:
            print("can load word embeddings on decoder side, tying weights")
            self.emit.data = self.inp_embedding.weight.data


class HMM_VI_Marginalized(HMM_VI_Layers):
    """
    This version of the model marginalizes the loss at each step instead of relying on a sample - which is much worse
    The math isn't worked out yet (no guarantee that this is a proper lower bound), but it should be better at least
    """

    def forward(self, input, args, n_particles, test=False):
        T = nn.Softmax(dim=0)(self.T)  # NOTE: not in log-space
        pi = nn.Softmax(dim=0)(self.pi)
        emit = self.calc_emit()

        # run the input and teacher-forcing inputs through the embedding layers here
        seq_len, batch_sz = input.size()
        emb = self.inp_embedding(input)
        hidden = self.init_hidden(batch_sz, self.nhid, 2)  # bidirectional
        hidden_states, (_, _) = self.encoder(emb, hidden)
        hidden_states = hidden_states.repeat(1, n_particles, 1)

        # run the z-decoder at this point, evaluating the NLL at each step
        z = self.init_hidden(batch_sz * n_particles, self.z_dim, squeeze=True)

        nlls = hidden_states.data.new(seq_len, batch_sz * n_particles)
        loss = 0

        # now a log-prob
        prior_probs = pi.unsqueeze(0).expand(batch_sz * n_particles, self.z_dim)

        logits = self.init_hidden(batch_sz * n_particles, self.z_dim, squeeze=True)

        for i in range(seq_len):
            # logits = self.logits(torch.cat([hidden_states[i], h], 1))
            # logits = self.logits(nn.functional.relu(self.z_decoder(hidden_states[i], logits)))
            logits = self.logits(nn.functional.relu(self.z_decoder(torch.cat([hidden_states[i], z], 1), logits)))

            # build the next z sample
            if test:
                q = OneHotCategorical(logits=logits)
                z = q.sample()
            else:
                q = RelaxedOneHotCategorical(temperature=self.temp, logits=logits)
                z = q.rsample()

            lse = log_sum_exp(logits, dim=1).view(-1, 1)
            log_probs = logits - lse

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

        # now, we calculate the final log-marginal estimator
        return loss.sum(), nlls.sum(), (seq_len * batch_sz * n_particles), 0

    # we need to fix evaluate to use the pfilter evaluation
    def evaluate(self, data_source, args, num_importance_samples=3):
        self.eval()
        old_anneal = args.anneal  # save to replace after evaluation
        args.anneal = 1.
        total_loss = 0
        total_log_marginal = 0
        total_resamples = 0
        batch_idx = 0
        total_tokens = 0

        for batch in data_source:
            if args.cuda:
                batch = batch.cuda()
            batch = batch.squeeze(0).t().contiguous()
            data = Variable(batch)  # squeeze for 1 billion
            elbo, _, _, resamples = super(HMM_VI_Marginalized, self).forward(data, args, num_importance_samples, test=True)
            total_log_marginal += self.eval_log_marginal(batch).sum()
            total_loss += elbo.sum().detach().data
            total_tokens += (data.size()[0] * data.size()[1])
            total_resamples += resamples
            batch_idx += 1

        if args.dataset != '1billion':
            total_tokens = 1
        args.anneal = old_anneal

        return total_loss[0] / total_tokens, total_log_marginal / total_tokens


class HMM_MFVI(HMM_VI_Layers):
    """
    This model assumes that q(z) = \prod_i q(z_i)
    """
    def __init__(self, *args, **kwargs):
        super(HMM_MFVI, self).__init__(*args, **kwargs)
        self.logits = nn.Linear(2 * self.nhid, self.z_dim)

    def forward(self, input, args, n_particles, test=False):
        """
        n_particles is interpreted as 1 for now to not screw anything up
        """
        n_particles = 1
        T = nn.Softmax(dim=0)(self.T)  # NOTE: not in log-space
        pi = nn.Softmax(dim=0)(self.pi)
        emit = self.calc_emit()

        # run the input and teacher-forcing inputs through the embedding layers here
        seq_len, batch_sz = input.size()
        emb = self.inp_embedding(input)
        hidden = self.init_hidden(batch_sz, self.nhid, 2)  # bidirectional
        hidden_states, (_, _) = self.encoder(emb, hidden)
        hidden_states = hidden_states.repeat(1, n_particles, 1)

        # run the z-decoder at this point, evaluating the NLL at each step
        z = self.init_hidden(batch_sz * n_particles, self.z_dim, squeeze=True)

        nlls = hidden_states.data.new(seq_len, batch_sz * n_particles)
        loss = 0

        # now a log-prob
        prior_probs = pi.unsqueeze(0).expand(batch_sz * n_particles, self.z_dim)

        logits = self.init_hidden(batch_sz * n_particles, self.z_dim, squeeze=True)

        for i in range(seq_len):
            # logits = self.logits(torch.cat([hidden_states[i], h], 1))
            # logits = self.logits(nn.functional.relu(self.z_decoder(hidden_states[i], logits)))
            logits = self.logits(hidden_states[i])

            # build the next z sample
            q = RelaxedOneHotCategorical(temperature=Variable(torch.Tensor([args.temp]).cuda()), logits=logits)
            z = q.sample()

            lse = log_sum_exp(logits, dim=1).view(-1, 1)
            log_probs = logits - lse

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

        # now, we calculate the final log-marginal estimator
        return loss.sum(), nlls.sum(), (seq_len * batch_sz * n_particles), 0


class HMM_MFVI_Yoon(HMM_MFVI):
    """
    This is a particularly clever way to compute the ELBO for the full HMM at the same time
    we assume q(z) = \prod q(z_t)

    ELBO = E_q[log p(x, z)] + H[q]
         = \sum_t \sum_{z_t} q(z_t) (\log p(x_t | z_t) + \ (unary)
           \sum_{t}\sum_{z_t, z_{t - 1}} q(z_t)q(z_{t - 1}) \log p(z_t|z_{t - 1}) + \ (binary)
           \sum_{z_1} q(z_1) \log p(z_1)

    Note that this is the mean field model, the corresponding factored version is below
    """
    def forward(self, input, args, n_particles, test=False):
        """
        If n_particles != 1, this the IWAE estimator, which doesn't make sense here
        """
        n_particles = 1
        T = F.log_softmax(self.T, 0)  # NOTE: in log-space
        pi = F.log_softmax(self.pi, 0)  # in log-space, intentionally
        emit = self.calc_emit()  # also in log-space

        seq_len, batch_sz = input.size()
        emb = self.inp_embedding(input)
        hidden = self.init_hidden(batch_sz, self.nhid, 2)  # bidirectional
        hidden_states, (_, _) = self.encoder(emb, hidden)
        hidden_states = hidden_states.repeat(1, n_particles, 1)

        elbo = 0
        NLL = 0

        # now a logit
        prior_logits = pi.unsqueeze(0).expand(batch_sz * n_particles, self.z_dim)

        prev_probs = None

        for i in range(seq_len):
            logits = F.log_softmax(self.logits(hidden_states[i]), 1)  # log q(z_i)
            probs = logits.exp()  # q(z_i)
            emission = F.embedding(input[i].repeat(n_particles), emit)  # log p(x_i | z_i)

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

        if n_particles != 1:
            elbo = log_sum_exp(elbo.view(n_particles, batch_sz), 0) - math.log(n_particles)
            NLL = NLL.view(n_particles, batch_sz).mean(0)

        # now, we calculate the final log-marginal estimator
        return -elbo.sum(), NLL.sum(), (seq_len * batch_sz), 0


class HMM_MFVI_Yoon_Deep(HMM_MFVI_Yoon):
    """
    Same as above, but with a more complex map from the LSTM states to the logits
    """
    def __init__(self, *args, **kwargs):
        super(HMM_MFVI_Yoon_Deep, self).__init__(*args, **kwargs)
        self.z_decoder = None
        self.logits = nn.Sequential(nn.Linear(2 * self.nhid, self.nhid),
                                    nn.ReLU(),
                                    nn.Linear(self.nhid, self.z_dim))

class HMM_MFVI_Mine(HMM_MFVI_Yoon_Deep):
    """
    This model does something different - it computes two losses: log p(x), exactly computed using
    the forward-backward algorithm, and KL(p(z|x) || q(z)) - i.e. it tries to fit the approximate posterior
    to the true posterior. Why do this instead of using the exact posterior everywhere? Because we think
    downstream we can optimize the approximate posterior.
    """

    def __init__(self, *args, **kwargs):
        super(HMM_MFVI_Mine, self).__init__(*args, **kwargs)
        # self.encoder = torch.nn.LSTM(self.word_dim, self.nhid, 2, dropout=0, bidirectional=True)

    def forward_backward(self, input, stop=False):
        """
        Modify the forward-backward to compute beta[t], since we need that
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

        # keep around for now, but unnecessary in our models
        for t in range(seq_len - 2, -1, -1):
            beta[t] = log_sum_exp(T.unsqueeze(0) +
                                  beta[t + 1].unsqueeze(2) +
                                  F.embedding(input[t + 1], emit).unsqueeze(2), 1)

        log_marginal = log_sum_exp(alpha[-1] + beta[-1], dim=-1)

        return [alpha[i] + beta[i] - log_marginal.unsqueeze(1) for i in range(seq_len)], 0, log_marginal

    def forward(self, input, args, n_particles, test=False):
        if test:
            return super(HMM_MFVI_Mine, self).forward(input, args, n_particles, test)

        seq_len, batch_sz = input.size()
        emb = self.inp_embedding(input)
        hidden = self.init_hidden(batch_sz, self.nhid, 2)  # bidirectional
        hidden_states, (_, _) = self.encoder(emb, hidden)

        log_posterior, _, log_marginal = self.forward_backward(input, stop=not test)

        KL = 0
        for i in range(seq_len):
            log_post = Variable(log_posterior[i].data)
            logits = F.log_softmax(self.logits(hidden_states[i]), 1)  # log q(z_i)
            KL += (logits.exp() * (logits - log_post)).sum(1)

        loss = -log_marginal + KL

        # now, we calculate the final log-marginal estimator
        return loss.sum(), -log_marginal.data.sum(), (seq_len * batch_sz), 0
