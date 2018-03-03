"""
For sake of time, I've copied a bunch of code from HMM to HMM_EM;
TODO: mv HMM_EM -> HMM, and add some Variable calls around to patch things up
"""

import torch
from torch.autograd import Variable
import pdb  # noqa: F401
from torch import nn
import numpy as np
from utils import log_sum_exp, any_nans


class HMM(nn.Module):
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
        return marginals[0].log()

    def generate_data(self, N, T):
        # TODO (then we can drop the hmmlearn package except in tests)
        pass


class HMM_EM(nn.Module):
    def __init__(self, z_dim, x_dim):
        super(HMM_EM, self).__init__()
        self.T = nn.Parameter(torch.Tensor(z_dim, z_dim))  # transition matrix -> each column is normalized
        self.pi = nn.Parameter(torch.zeros(z_dim))  # initial likelihoods - real probabilities
        self.emit = nn.Parameter(torch.Tensor(x_dim, z_dim))  # takes a 1-hot Z, and turns it into an x sample

        self.z_dim = z_dim
        self.x_dim = x_dim

        self.randomly_initialize()

    def forward_backward(self, input):
        """
        input: Variable([seq_len x batch_size])
        """
        input = input.long()

        seq_len, batch_size = input.size()
        alpha = [None for i in range(seq_len)]
        beta = [None for i in range(seq_len)]

        T = nn.Softmax(dim=0)(self.T)
        pi = nn.Softmax(dim=0)(self.pi)
        emit = nn.Softmax(dim=0)(self.emit)

        # forward pass
        alpha[0] = emit[input[0]] * pi.view(1, -1)
        beta[-1] = Variable(torch.ones(batch_size, self.z_dim))
        if T.is_cuda:
            beta[-1] = beta[-1].cuda()

        for t in range(1, seq_len):
            alpha[t] = (emit[input[t]] * torch.mm(alpha[t - 1], T.t()))

        for t in range(seq_len - 2, -1, -1):
            beta[t] = torch.mm((emit[input[t + 1]] * beta[t + 1]), T)

        log_marginal = log_sum_exp(alpha[0].log() + beta[0].log(), dim=-1)
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

    def evaluate(self, data_source, args, num_samples=None):
        self.eval()
        total_loss = 0
        for batch in data_source:
            if args.cuda:
                batch = batch.cuda()
            data = Variable(batch.squeeze().t().contiguous())  # squeeze for 1 billion
            loss = self.forward(data)
            loss = loss.sum()
            total_loss += loss.detach().data
        return total_loss[0], total_loss[0]

    def train_epoch(self, train_data, optimizer, epoch, args, num_samples=None):
        self.train()
        total_loss = 0
        for batch in train_data:
            if args.cuda:
                batch = batch.cuda()
            data = Variable(batch.squeeze().t().contiguous())  # squeeze for 1 billion
            optimizer.zero_grad()
            loss = self.forward(data)
            if any_nans(loss):
                pdb.set_trace()
            else:
                loss = loss.sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach()
        return total_loss.data[0], -1
