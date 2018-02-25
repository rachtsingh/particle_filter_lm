import torch
import pdb  # noqa: F401
from torch import nn
from torch.distributions import OneHotCategorical


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
        # pass here
        x = torch.zeros((T, N, self.x_dim)).long()
        x[0] = OneHotCategorical(probs=self.pi.data).sample(())
        for t in range(1, T):
            pass
