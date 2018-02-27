import numpy as np
from matplotlib import pyplot as plt

def filter():
    colors = ['green', 'blue', 'red']
    exact = [-3408.98059082, -3443.19781494, -3440.65063477]
    ax = plt.gca()
    w_filter = np.load("real_data_w_filter.npy")
    for i, z_dim in enumerate([5, 15, 30]):
        data_points = np.zeros(3)
        for j, n_particles in enumerate([5, 15, 30]):
            data_points[j] = w_filter[i][j]
        ax.plot(np.array([5, 15, 30]), data_points, 'o', color=colors[i], label='w/ filter {:<2}-dim'.format(z_dim))
        ax.plot(np.array([5, 15, 30]), np.array([exact[i]]).repeat(3), 
                label='exact marginal', color=colors[i])

    plt.legend(loc=4)
    plt.xlabel('n_particles')
    plt.ylabel('log p(x)')
    ax.set_xlim([0, 35])
    plt.savefig('filters_on_hmm.png', dpi=300)

def no_filter():
    colors = ['green', 'blue', 'red']
    exact = [-3408.98059082, -3443.19781494, -3440.65063477]
    ax = plt.gca()
    w_filter = np.load("real_data_wo_filter.npy")
    for i, z_dim in enumerate([5, 15, 30]):
        data_points = np.zeros(3)
        for j, n_particles in enumerate([5, 15, 30]):
            data_points[j] = w_filter[i][j]
        ax.plot(np.array([5, 15, 30]), data_points, 'o', color=colors[i], label='w/o filter {:<2}-dim'.format(z_dim))
        ax.plot(np.array([5, 15, 30]), np.array([exact[i]]).repeat(3), 
                label='exact marginal', color=colors[i])

    plt.legend(loc=4)
    plt.xlabel('n_particles')
    plt.ylabel('log p(x)')
    ax.set_xlim([0, 35])
    plt.savefig('no_filter_on_hmm.png', dpi=300)

def plot_together():
    colors = ['green', 'blue', 'red']
    exact = [-3408.98059082, -3443.19781494, -3440.65063477]
    ax = plt.gca()
    w_filter = np.load("real_data_w_filter.npy")
    for i, z_dim in enumerate([5, 15, 30]):
        data_points = np.zeros(3)
        for j, n_particles in enumerate([5, 15, 30]):
            data_points[j] = w_filter[i][j]
        ax.plot(np.array([5, 15, 30]), data_points, 'o', color=colors[i], label='w/ filter {:<2}-dim'.format(z_dim))
        ax.plot(np.array([5, 15, 30]), np.array([exact[i]]).repeat(3), 
                label='exact marginal', color=colors[i])

    wo_filter = np.load("real_data_wo_filter.npy")
    for i, z_dim in enumerate([5, 15, 30]):
        data_points = np.zeros(3)
        for j, n_particles in enumerate([5, 15, 30]):
            data_points[j] = wo_filter[i][j]
        ax.plot(np.array([5, 15, 30]), data_points, 'o', mfc='none', mec=colors[i], label='w/o filter {:<2}-dim'.format(z_dim))
        ax.plot(np.array([5, 15, 30]), np.array([exact[i]]).repeat(3), 
                label='exact marginal', color=colors[i])

    plt.legend(loc=4)
    plt.xlabel('n_particles')
    plt.ylabel('log p(x)')
    ax.set_xlim([0, 35])
    plt.savefig('together.png', dpi=300)

plot_together()