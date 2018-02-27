from subprocess import check_output
import numpy as np
import pdb

values = np.zeros((3, 3))
true_marginal = 0

for i, z_dim in enumerate([150]):
    for j, n_particles in enumerate([5, 25, 50]):
        out = check_output("python run_hmm_exp.py --epochs 20 --nhid 32 --batch_size 10 --quiet --print-best --temp 0.8 --temp_prior 0.5 --z-dim {} --num-importance-samples {} --x-dim 100".format(z_dim, n_particles).split(' '))
        v, true_marginal = map(float, out.split(","))
        values[i][j] = v
        print(z_dim, n_particles, v, true_marginal)
# np.save('real_data_wo_filter.npy', values)
pdb.set_trace()
