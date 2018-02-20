import os
import pdb  # noqa: F401
from torchtext import data


class WordLevelSST(data.Dataset):

    urls = ['http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip']
    dirname = 'trees'
    name = 'sst'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, subtrees=False,
                 fine_grained=False, **kwargs):
        """Create an SST dataset instance given a path and fields.

        Arguments:
            path: Path to the data file
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            subtrees: Whether to include sentiment-tagged subphrases
                in addition to complete examples. Default: False.
            fine_grained: Whether to use 5-class instead of 3-class
                labeling. Default: False.
            Rema
            ining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field), ('target', text_field), ('label', label_field)]

        with open(os.path.expanduser(path)) as f:
            from nltk.tree import Tree
            examples = []
            for line in f:
                tree = Tree.fromstring(line)
                text = text_field.preprocess(' '.join(tree.flatten().leaves())) + ['<eos>']
                target = ['<bos>'] + text[:-1]
                labels = [int(t.label()) for t in tree.subtrees(lambda t: t.height() == 2)]
                examples.append(data.Example.fromlist([text, target, labels], fields))

        super(WordLevelSST, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, root='./data',
               train='train.txt', validation='dev.txt', test='test.txt',
               train_subtrees=False, **kwargs):
        """Create dataset objects for splits of the SST dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            validation: The filename of the validation data, or None to not
                load the validation set. Default: 'dev.txt'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'test.txt'.
            train_subtrees: Whether to use all subtrees in the training set.
                Default: False.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        path = os.path.join(cls.download(root), cls.dirname)

        train_data = None if train is None else cls(
            os.path.join(path, train), text_field, label_field, subtrees=train_subtrees,
            **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), text_field, label_field, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), text_field, label_field, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)

    @classmethod
    def iters(cls, batch_size=32, device=0, root='./data', vectors=None, **kwargs):
        """Creater iterator objects for splits of the SST dataset.

        Arguments:
            batch_size: Batch_size
            device: Device to create batches on. Use - 1 for CPU and None for
                the currently active GPU device.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            vectors: one of the available pretrained vectors or a list with each
                element one of the available pretrained vectors (see Vocab.load_vectors)
            Remaining keyword arguments: Passed to the splits method.
        """

        TEXT = data.Field()
        LABEL = data.Field(use_vocab=False)

        train, val, test = cls.splits(TEXT, LABEL, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)
        LABEL.build_vocab(train)

        return data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device)
