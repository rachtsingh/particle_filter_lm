"""
This file is really just a visualization script; it'll change often to produce whatever plot is necessary
"""
import numpy as np
from matplotlib import pyplot as plt


def main():
    plt.figure(figsize=(5, 5))
    files = [('true_eval.npy', 'oracle log p(x)'),
             ('vi_traj_eval.npy', 'VI log p(x) (no-filter)'),
             ('vi_filter_traj_eval.npy', 'VI log p(x) (filter)'),
             ('em_traj_2_eval.npy', 'EM log p(x)'),
             ('elbo_vi.npy', 'VI ELBO'),
             ('fivo_vi.npy', 'VI FIVO')]
    for f, label in files:
        data = np.load(f)
        x = np.arange(data.shape[0])
        plt.plot(x, data, label=label, linewidth=1)
    plt.legend(loc=4)
    plt.ylim([-14800, -14625])
    plt.xlabel('epochs')
    plt.ylabel('log marginal estimate')
    plt.savefig('misspecified.png', dpi=300)


if __name__ == '__main__':
    main()
