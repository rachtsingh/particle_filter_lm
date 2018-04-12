"""Build the vocabulary from corpus

This file is forked from pytorch/text repo at Github.com"""
import os
import pickle
import logging
from collections import defaultdict, Counter

from tqdm import tqdm
logger = logging.getLogger(__name__)


def _default_unk_index():
    return 0

class Vocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.
    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        word2idx: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        idx2word: A list of token strings indexed by their numerical identifiers.
    """
    def __init__(self, counter, max_size=None, min_freq=1):
        """Create a Vocab object from a collections.Counter.
        Arguments:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
        """
        self.freqs = counter
        self.max_size = max_size
        self.min_freq= min_freq
        self.specials = ['<unk>', '<s>']
        self.build()


    def build(self):
        """Build the required vocabulary according to attributes

        We need an explicit <unk> for NCE because this improve the precision of
        word frequency estimation in noise sampling
        """
        counter = self.freqs.copy()
        min_freq = max(self.min_freq, 1)

        self.idx2word = list(self.specials)

        # Do not count the BOS and UNK as frequency term
        for word in self.specials:
            del counter[word]
        max_size = None if self.max_size is None else self.max_size + len(self.idx2word)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        unk_freq = 0
        for word, freq in words_and_frequencies:
            if freq < min_freq:
                # count the unk frequency
                unk_freq += freq
                continue
            if len(self.idx2word) == max_size:
                continue
            self.idx2word.append(word)

        self.word2idx = defaultdict(_default_unk_index)
        self.word2idx.update({
            word: idx for idx, word in enumerate(self.idx2word)
        })

        self.idx2count = [self.freqs[word] for word in self.idx2word]
        self.idx2count[0] += unk_freq
        self.idx2count[1] = 0

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.word2idx != other.word2idx:
            return False
        if self.idx2word != other.idx2word:
            return False
        return True

    def __len__(self):
        return len(self.idx2word)

    def extend(self, v, sort=False):
        words = sorted(v.idx2word) if sort else v.idx2word
        #TODO: speedup the dependency
        for w in words:
            if w not in self.word2idx:
                self.idx2word.append(w)
                self.word2idx[w] = len(self.idx2word) - 1

    def write_freq(self, freq_file):
        """Write the word-frequency pairs into text file"""
        with open(freq_file, 'w') as f:
            for word, freq in self.freqs.most_common():
                f.writelines('{} {}\n'.format(word, freq))


def check_vocab(vocab):
    """A util function to check the vocabulary correctness"""
    ## one word for one index
    assert len(vocab.idx2word) == len(vocab.word2idx)

    # no duplicate words in idx2word
    assert len(set(vocab.idx2word)) == len(vocab.idx2word)

def get_vocab(base_path, file_list, max_size=10002, force_recount=False):
    """Build vocabulary file with each line the word and frequency

    The vocabulary object is cached at the first build, aiming at reducing
    the time cost for pre-process during training large dataset

    Args:
        - sentences: sentences with BOS and EOS
        - min_freq: minimal frequency to truncate
        - force_recount: force a re-count of word frequency regardless of the
        Count cache file

    Return:
        - vocab: the Vocab object
    """
    counter = Counter()
    cache_file = os.path.join(base_path, 'vocab.pkl')

    if os.path.exists(cache_file) and not force_recount:
        logger.debug('Load cached vocabulary object')
        vocab = pickle.load(open(cache_file, 'rb'))
        if max_size is not None:
            vocab.max_size = max_size
        vocab.build()
        logger.debug('Load cached vocabulary object finished')
    else:
        logger.debug('Refreshing vocabulary')
        for filename in file_list:
            full_path = os.path.join(base_path, filename)
            for line in tqdm(open(full_path, 'r'), desc='Building vocabulary: '):
                counter.update(line.split())
                counter.update(['<s>', '</s>'])
        vocab = Vocab(counter, max_size=max_size)
        vocab.build()
        logger.debug('Refreshing vocabulary finished')

        # saving for future uses
        freq_file = os.path.join(base_path, 'freq.txt')
        vocab.write_freq(freq_file)
        pickle.dump(vocab, open(cache_file, 'wb'))

    check_vocab(vocab)
    return vocab
