from torch.autograd import Variable
import sys
import subprocess


def get_sha():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def print_in_epoch_summary(epoch, batch_idx, batch_size, dataset_size, loss, NLL,
                           KLs, tokens, msg):
    """
    loss: ELBO/IWAE/other final loss, *divided by `tokens`*
    """
    kl_string = '\t'.join(["KL({}): {:.3f}".format(key, val) for key, val in KLs.items()])
    print('Train Epoch: {} [{:<5}/{} ({:<2.0f}%)]\tLoss: {:.3f}\tNLL: {:.3f}\t{}'.format(
        epoch, (batch_idx + 1) * batch_size, dataset_size,
        100. * (batch_idx + 1) / (dataset_size / batch_size),
        loss,
        NLL,
        kl_string))
    sys.stdout.flush()
