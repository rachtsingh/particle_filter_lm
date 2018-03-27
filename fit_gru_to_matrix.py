import torch
import sys
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.distributions import OneHotCategorical

WORD_DIM = 300
Z_HID = 50
Z_EMB = 50


def main():
    params = torch.load(sys.argv[1])
    T = F.log_softmax(Variable(params['T']), 0)
    z_dim = T.size(0)
    gru = nn.GRUCell(WORD_DIM + Z_EMB, Z_HID).cuda()
    z_emb = nn.Linear(z_dim, Z_EMB).cuda()
    project = nn.Linear(Z_HID, z_dim).cuda()
    optimizer = torch.optim.SGD(gru.parameters(), lr=0.01)

    for i in range(1, 5000):
        # construct a batch of z_dim datapoints (one for each possible value of z)
        # and the hidden states of the GRU at that point - which we need to ignore (i.e. robust to noise)
        states = Variable(torch.randn(z_dim, Z_HID).cuda())
        z = Variable(torch.eye(z_dim).cuda())

        y = T.t()
        pred = F.log_softmax(project(gru(torch.cat([Variable(torch.zeros(z_dim, WORD_DIM).cuda()), z_emb(z)], 1), states)), 1)
        error = (pred.exp() * (pred - y)).sum()
        regularization = 0.001 * sum([p.pow(2).sum() for p in gru.parameters()])
        (error + regularization).backward()
        optimizer.step()

        if i % 100 == 0:
            print("({:.3f}, {:.3f})".format(error.data[0], regularization.data[0]))

    with open('gru_2.pt', 'w') as f:
        torch.save((gru, z_emb, project), f)


if __name__ == '__main__':
    main()
