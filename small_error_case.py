import torch
import torch.nn as nn
from torch.autograd import Variable
import sys

INPUT_SIZE = 2048
HIDDEN_SIZE = 1024


class IndexingIssues(nn.Module):
    def __init__(self):
        super(IndexingIssues, self).__init__()
        self.enc = nn.LSTMCell(INPUT_SIZE, HIDDEN_SIZE)

    def forward(self, input):
        seq_len, batch_sz, _ = input.size()
        h = Variable(input.data.new(batch_sz, HIDDEN_SIZE).zero_())
        c = Variable(input.data.new(batch_sz, HIDDEN_SIZE).zero_())
        for i in range(seq_len):
            h, c = self.enc(input[i], (h, c))
            # idx = Variable(torch.arange(batch_sz).long().cuda().view(-1))  # just identity for now
            # h = torch.index_select(h, 0, idx)
            # c = torch.index_select(c, 0, idx)
            idx = torch.arange(batch_sz).long().cuda().view(-1)  # just identity for now
            h = h[idx]
            c = c[idx]
        return h.sum()


def main():
    output_name = sys.argv[1]
    data = Variable(torch.randn(15, 160, INPUT_SIZE).cuda())
    model = IndexingIssues().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    with torch.autograd.profiler.profile() as prof:
        for i in range(10):
            optimizer.zero_grad()
            loss = model(data)
            loss.backward()
            optimizer.step()
    prof.export_chrome_trace(output_name)


if __name__ == '__main__':
    main()
