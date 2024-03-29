"""
For sake of time, I've copied a bunch of code from HMM to HMM_EM;
kODO: mv HMM_EM -> HMM, and add some Variable calls around to patch things up
"""

import torch
from torch.autograd import Variable
import pdb  # noqa: F401
from torch import nn
from torch.nn import functional as F
import numpy as np
from src.utils import log_sum_exp, any_nans

SMALL = 1e-16


class HMM(nn.Module):
    """
    DEPRECATED
    """
    def __init__(self, z_dim, x_dim, params=None):
        super(HMM, self).__init__()
        self.T = torch.Tensor(z_dim, z_dim)  # transition matrix -> each column is normalized
        self.pi = torch.zeros(z_dim)  # initial likelihoods - real probabilities
        self.emit = torch.Tensor(x_dim, z_dim)  # takes a 1-hot Z, and turns it into an x sample - real probabilities

        if not(params is None):
            T, pi, emit = params
            self.T = torch.Tensor(T.T)
            self.pi = torch.Tensor(pi)
            self.emit = torch.Tensor(emit.T)

        self.z_dim = z_dim
        self.x_dim = x_dim

    def cuda(self, *args, **kwargs):
        if type(self.T) in (Variable, nn.Parameter):
            self.T.data = self.T.data.cuda()
            self.pi.data = self.pi.data.cuda()
            self.emit.data = self.emit.data.cuda()
        else:
            self.T = self.T.cuda()
            self.pi = self.pi.cuda()
            self.emit = self.emit.cuda()
        super(HMM, self).cuda(*args, **kwargs)

    def forward(self):
        pass

    def forward_backward(self, input):
        """
        input: [seq_len x batch_size]
        """
        input = input.long()

        seq_len, batch_size = input.size()
        alpha = torch.zeros((seq_len, batch_size, self.z_dim))
        beta = torch.zeros((seq_len, batch_size, self.z_dim))
        if self.T.is_cuda:
            alpha, beta = alpha.cuda(), beta.cuda()

        # forward pass
        alpha[0] = (self.emit[input[0]] * self.pi.view(1, -1))
        beta[seq_len - 1, :, :] = 1.

        for t in range(1, seq_len):
            alpha[t] = (self.emit[input[t]] * torch.mm(alpha[t - 1], self.T.t()))

        for t in range(seq_len - 2, -1, -1):
            beta[t] = torch.mm((self.emit[input[t + 1]] * beta[t + 1]), self.T)

        normalization = (alpha * beta).sum(-1, keepdim=True)
        posterior_marginal = (alpha * beta) / normalization

        return alpha, beta, posterior_marginal

    def log_marginal(self, input):
        """
        input: [seq_len x batch_size]
        """
        alpha, beta, _ = self.forward_backward(input)
        marginals = (alpha * beta).sum(-1)
        assert max(marginals.var(0).abs() < 1e-6)
        print(marginals[0])
        return marginals[0].log()

    def eval_log_marginal(self, input):
        """
        eval_log_marginal always takes the signature of input: Tensor -> Tensor
        """
        return self.log_marginal(input)

    def generate_data(self, N, T):
        # TODO (then we can drop the hmmlearn package except in tests)
        pass


class HMM_EM(nn.Module):
    """
    This is a log-value optimized implementation of the HMM fit via EM, with a separated out observation model
    """
    def __init__(self, z_dim, x_dim):
        super(HMM_EM, self).__init__()
        self.T = nn.Parameter(torch.Tensor(z_dim, z_dim))  # transition matrix -> each column is normalized
        self.pi = nn.Parameter(torch.zeros(z_dim))  # initial likelihoods - real probabilities
        self.emit = nn.Parameter(torch.Tensor(x_dim, z_dim))  # takes a 1-hot Z, and turns it into an x sample

        self.z_dim = z_dim
        self.x_dim = x_dim

        self.randomly_initialize()

    def set_params(self, params):
        T, pi, emit, hidden = params
        self.T.data = T
        self.pi.data = pi
        self.emit.data = emit
        if hidden is not None:
            self.hidden.data = hidden

    def log_prob(self, input, precompute=None):
        """
        Returns a [batch_size x z_dim] log-probability of input given state z
        """
        emit_prob, = precompute
        return F.embedding(input, emit_prob)

    def calc_emit(self):
        """
        Amortize calculation of the emisssion
        """
        return F.log_softmax(self.emit, 0)

    def forward_backward(self, input):
        """
        input: Variable([seq_len x batch_size])
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
        # for t in range(seq_len - 2, -1, -1):
        #     beta_expand = beta[t + 1].unsqueeze(1).expand(batch_size, self.z_dim, self.z_dim)
        #     beta[t] = log_sum_exp(beta_expand + T.t().unsqueeze(0), 2) + emit[input[t + 1]]

        log_marginal = log_sum_exp(alpha[-1] + beta[-1], dim=-1)

        return alpha, beta, log_marginal

    def randomly_initialize(self):
        T = np.random.random(size=(self.z_dim, self.z_dim))
        T = T/T.sum(axis=1).reshape((self.z_dim, 1))

        pi = np.random.random(size=(self.z_dim,))
        pi = pi/pi.sum()

        emit = np.random.random(size=(self.z_dim, self.x_dim))
        emit = emit/emit.sum(axis=1).reshape((self.z_dim, 1))

        self.T.data = torch.from_numpy(T.T).log()
        self.pi.data = torch.from_numpy(pi).log()
        self.emit.data = torch.from_numpy(emit.T).log()
        self.float()

    def forward(self, input):
        return -self.log_marginal(input)

    def log_marginal(self, input):
        """
        input: [seq_len x batch_size]
        """
        _, _, log_marginal = self.forward_backward(input)
        return log_marginal

    def eval_log_marginal(self, input):
        return self.log_marginal(Variable(input)).data

    def evaluate(self, data_source, args, num_samples=None):
        self.eval()
        total_loss = 0
        total_tokens = 0
        for batch in data_source:
            if args.cuda:
                batch = batch.cuda()
            data = Variable(batch.squeeze(0).t().contiguous())  # squeeze for 1 billion
            total_tokens += (data.size()[0] * data.size()[1])
            loss = self.forward(data)
            loss = loss.sum()
            total_loss += loss.detach().data

        if args.dataset not in ('1billion', 'ptb'):
            total_tokens = 1

        return total_loss[0] / float(total_tokens), -total_loss[0] / float(total_tokens)

    def train_epoch(self, train_data, optimizer, epoch, args, num_samples=None):
        self.train()
        total_loss = 0
        print(len(train_data))

        for i, batch in enumerate(train_data):
            if (i + 1) % 100 == 0:
                print(i + 1)
            if args.cuda:
                batch = batch.cuda()
            data = Variable(batch.squeeze(0).t().contiguous())  # squeeze for 1 billion
            optimizer.zero_grad()
            loss = self.forward(data)
            loss = loss.sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach()
        return total_loss.data[0], -1


class HMM_EM_Layers(HMM_EM):
    """
    This model has a more thick observation model, so it can be more powerful
    """

    def __init__(self, z_dim, x_dim, hidden_size):
        super(HMM_EM_Layers, self).__init__(z_dim, x_dim)

        self.emit = nn.Parameter(torch.zeros(x_dim, hidden_size))
        self.hidden = nn.Parameter(torch.zeros(hidden_size, z_dim))

        self.hidden_size = hidden_size

        # fix the random initialization
        self.emit.data.uniform_(-0.01, 0.01)
        self.hidden.data.uniform_(-0.01, 0.01)

    def calc_emit(self):
        return F.log_softmax(torch.mm(self.emit, self.hidden), 0)

    def load_embedding(self, embedding):
        x_dim, word_dim = embedding.size()
        if x_dim != self.x_dim or word_dim != self.hidden_size:
            raise ValueError("embedding has size: {} when expected: ({}, {})".format(embedding.size(), self.x_dim, self.hidden_size))
        self.emit.data = embedding
        self.emit.data[:4] = torch.randn(4, self.hidden_size)
