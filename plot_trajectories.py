"""
This file is really just a visualization script; it'll change often to produce whatever plot is necessary
"""
import numpy as np
import sys
from matplotlib import pyplot as plt


def main():
    files = sys.argv[1:]
    print(files)
    for f in files:
        data = np.load(f)
        x = np.arange(data.shape[0])
        plt.plot(x, data)
    plt.show()


if __name__ == '__main__':
    main()
