"""
Just a few changes to make things more flexible
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5, batch_dim=0):
        if not self.training or not dropout:
            return x
        s = list(x.size())
        s[batch_dim] = 1
        m = x.data.double().new(*s).bernoulli_(1 - dropout)
        mask = Variable(m.type(x.data.__class__), requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x
