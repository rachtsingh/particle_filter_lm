import os
import torch
import torchtext
import spacy
import re
import pdb

from torchtext.datasets import PennTreeBank, WikiText2
from torchtext import data

class Seq2SeqLMDataset(data.Dataset):
    def __init__(self, path, text_field, **kwargs):
        fields = [('text', text_field)]
        examples = []
        with open(path) as f:
            for line in f:
                text = text_field.preprocess(line) + ['<eos>']
                examples.append(data.Example.fromlist([text], fields))

        data.Dataset.__init__(self, examples, fields, **kwargs)
    
    @staticmethod
    def sort_key(ex):
        return len(ex.text)
    
    @classmethod
    def splits(cls, text_field, root='./data', **kwargs):
        return super(Seq2SeqLMDataset, cls).splits(root=root, text_field=text_field, **kwargs)

    @classmethod
    def iters(cls, batch_size=32, device=0, root='./data', vectors=None, **kwargs):
        """
        PTB, but with sorted batches
        """
        TEXT = data.Field()
        train, val, test = cls.splits(TEXT, root=root, **kwargs)
        TEXT.build_vocab(train, vectors=vectors)
        return data.BucketIterator.splits((train, val, test), batch_size=batch_size, device=device)

class PTBSeq2Seq(Seq2SeqLMDataset, PennTreeBank):
    pass

class WT2Seq2Seq(Seq2SeqLMDataset, WikiText2):
    pass

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
