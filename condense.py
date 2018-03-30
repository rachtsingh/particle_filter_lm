import torch
import numpy as np
import os
import pdb

NUM_EPOCHS = 22  # 1, ... 22
NUM_BATCHES = 305  # 0, 10, ... 3040

keys = ('exact_elbo', 'exact_marginal', 'sampled_elbo', 'sampled_iwae')


def main():
    data = {}
    for key in keys:
        data[key] = np.zeros((22 * 305, 8))

    for path in os.listdir('.'):
        if path[-3:] == '.pt':
            chunks = path.split('_')
            epoch = int(chunks[2])
            batch = int(chunks[3].split('.')[0])
            index = (epoch - 1) * NUM_BATCHES + (batch/10)
            nums = torch.load(path)
            for key in keys:
                data[key][index] = nums[key]

    np.savez("wo_filter_summary.npz", **data)

main()
