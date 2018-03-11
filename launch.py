import subprocess
import pdb

for temp_prior in [0.3, 0.4, 0.5]:
    for temp in [0.4, 0.5, 0.6, 0.8, 0.9]:
        with open("{}_{}.log".format(temp_prior, temp), 'w') as f:
            subprocess.call("python run_hmm_exp.py --inference vi --dataset 1billion --x-dim 20004 --lr 0.03 --batch_size 1 --model hmm_deep_vi --z-dim 100 --temp {} --temp_prior {} --hidden 150 --num-importance-samples 15 --nhid 128 --epochs 1".format(temp, temp_prior, temp_prior, temp).split(' '), stdout=f, stderr=subprocess.STDOUT)

pdb.set_trace()
