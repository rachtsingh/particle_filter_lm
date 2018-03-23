from torch.autograd import Variable
import sys
import subprocess
import numpy as np
import pdb  # noqa: F401


def show_memusage(device=0):
    import gpustat
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print("{}/{}".format(item["memory.used"], item["memory.total"]))


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
    kl_string = ' | '.join(["{}: {:.2f}".format(key, val) for key, val in KLs.items()])
    print('Train Epoch: {} [{:<5}/{} ({:<2.0f}%)]\tLoss: {:.3f}\tNLL: {:.3f}\t{}'.format(
        epoch, (batch_idx + 1) * batch_size, dataset_size,
        100. * (batch_idx + 1) / (dataset_size / batch_size),
        loss,
        NLL,
        kl_string))
    sys.stdout.flush()


def log_sum_exp(arr, dim=0):
    """Apply log-sum-exp to get IWAE loss. It's assumed that the samples vary along the `dim` axis"""
    if not isinstance(arr, Variable):
        A = arr.max(dim)[0].unsqueeze(dim)
    else:
        A = Variable(arr.max(dim)[0].data, requires_grad=False).unsqueeze(dim)
    return (A + (arr - A).exp().sum(dim, keepdim=True).log()).squeeze(dim)


def any_nans(a):
    if isinstance(a, Variable):
        a = a.data
    return np.isnan(a.cpu().numpy()).any()
