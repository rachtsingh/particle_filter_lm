import torch
import sys


def main():
    embedding_file = sys.argv[1]
    corpus = sys.argv[2]

    corpus_map = {}
    with open(corpus, 'r') as f:
        for line in f:
            key, idx = line.split(' ')
            corpus_map[key.strip()] = int(idx.strip())

    # assuming 20k corpus, each is 300d
    embedding = torch.zeros(20004, 300)
    with open(embedding_file, 'r') as f:
        for line in f:
            line = line.split()
            key = line[0]
            values = torch.Tensor(map(float, line[1:]))
            assert values.size()[0] == 300
            embedding[corpus_map[key]] = values

    import pdb
    pdb.set_trace()
    with open('fasttext.pt', 'wb') as f:
        torch.save(embedding, f)


main()
