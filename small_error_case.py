import torch
import torch.nn as nn
from torch.autograd import Variable

INPUT_SIZE = 1024
HIDDEN_SIZE = 512


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
            idx = torch.arange(batch_sz).long().cuda().view(-1)  # just identity for now
            h = h[idx]
            c = c[idx]
        return h.sum()


def main():
    data = Variable(torch.randn(15, 80, INPUT_SIZE).cuda())
    model = IndexingIssues().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    with torch.autograd.profiler.profile() as prof:
        for i in range(5):
            optimizer.zero_grad()
            loss = model(data)
            loss.backward()
            optimizer.step()
    prof.export_chrome_trace("repro.prof")


if __name__ == '__main__':
    main()
