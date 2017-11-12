import os
import torch
import torchtext

from torchtext.datasets import WikiText2
from torchtext.data import BucketIterator

class WT2Seq2Seq(WikiText2):
    @classmethod
    def iters(cls, batch_size=32, bptt_len=35, device=0, root='.data', wv_dir='.',
              wv_type=None, wv_dim='300d', **kwargs):
        """
        This is the same as the WikiText2 dataset, but we iterate over it completely differently
        """
        TEXT = data.Field()
        train, val, test = cls.splits(TEXT, root=root, **kwargs)
        TEXT.build_vocab(train, wv_dir=wv_dir, wv_type=wv_type, wv_dim=wv_dim)
        return data.BucketIterator.splits((train, val, test), batch_size=batch_size, device=device, sort_key=lambda ex: len(ex.text))

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
