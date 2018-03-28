import numpy as np
from matplotlib import pyplot as plt


files = ['hmm_exact_elbo.log', 'hmm_exact_marginal.log']  # , 'hmm_sampled_elbo.log', 'hmm_sampled_iwae.log']


def parse_path(path):
    data = []
    print(path)
    with open(path, 'r') as f:
        for line in f:
            if line[0] == '|':
                chunks = line.split('PPL')
                fake_ppl = chunks[1][2:].split('|')[0].strip()
                ppl = chunks[2][2:].strip()
                data.append(np.array([float(fake_ppl), float(ppl)]))
    data = np.stack(data)
    print(data)
    if path == files[0]:
        label = r'$\operatorname{ELBO}_{\on{exact}}$ -'
        return [(label + 'bound PPL', data[:, 0]),
                (label + 'true PPL', data[:, 1])]
    if path == files[1]:
        label = r'$\log p(x)$ - true PPL (sEM)'
        return [(label, data[:, 1])]


def main():
    plots = []
    for path in files:
        plots += parse_path(path)
    for label, arr in plots:
        


main()
