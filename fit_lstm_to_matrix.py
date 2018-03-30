import torch
import sys
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

WORD_DIM = 300
Z_HID = 50
Z_EMB = 50
REPEAT = 20


def main():
    params = torch.load(sys.argv[1])
    T = F.log_softmax(Variable(params['T']), 0)
    z_dim = T.size(0)
    lstm = nn.LSTMCell(WORD_DIM + Z_EMB, Z_HID).cuda()
    z_emb = nn.Linear(z_dim, Z_EMB).cuda()
    project = nn.Linear(Z_HID, z_dim).cuda()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)

    for i in range(1, 3000):
        # construct a batch of z_dim datapoints (one for each possible value of z)
        # and the hidden states of the GRU at that point - which we need to ignore (i.e. robust to noise)
        states = (Variable(torch.randn(REPEAT * z_dim, Z_HID).cuda()), Variable(torch.randn(REPEAT * z_dim, Z_HID).cuda()))
        z = Variable(torch.eye(z_dim).cuda().repeat(REPEAT, 1))

        y = T.t().repeat(REPEAT, 1)
        h, _ = lstm(torch.cat([Variable(torch.zeros(REPEAT * z_dim, WORD_DIM).cuda()), z_emb(z)], 1), states)
        pred = F.log_softmax(project(h), 1)
        error = (pred.exp() * (pred - y)).sum() / REPEAT
        regularization = 0.001 * sum([p.pow(2).sum() for p in lstm.parameters()] +
                                     [p.pow(2).sum() for p in z_emb.parameters()] +
                                     [p.pow(2).sum() for p in project.parameters()])
        (error + regularization).backward()
        optimizer.step()

        if i % 100 == 0:
            print("({:.3f}, {:.3f})".format(error.data[0], regularization.data[0]))

    with open('lstm_final_150.pt', 'w') as f:
        torch.save((lstm, z_emb, project), f)


if __name__ == '__main__':
    main()
