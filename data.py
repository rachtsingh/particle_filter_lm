from torchtext.datasets import PennTreeBank, WikiText2
from torchtext import data
import pdb  # noqa: F401


class Seq2SeqLMDataset(data.Dataset):
    def __init__(self, path, text_field, **kwargs):
        # pdb.set_trace()
        fields = [('text', text_field), ('target', text_field)]
        examples = []
        with open(path) as f:
            for line in f:
                text = text_field.preprocess(line) + ['<eos>']
                target = ['<bos>'] + text[:-1]
                examples.append(data.Example.fromlist([text, target], fields))

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
        return data.BucketIterator.splits((train, val, test), batch_size=batch_size, device=device, repeat=False)


class PTBSeq2Seq(Seq2SeqLMDataset, PennTreeBank):
    pass


class WT2Seq2Seq(Seq2SeqLMDataset, WikiText2):
    pass
