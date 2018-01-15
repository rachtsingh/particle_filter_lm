import subprocess

for nhid in [150, 300, 500]:
    for dropouti in [0.1, 0.3]:
        for dropoute in [0.1]:
            for lr in [0.01, 0.025, 0.1]:
                filename = "{}_{}_{}_{}.log".format(lr, nhid, dropouti, dropoute)
                command = "python main.py --epochs 25 --lr {} --emsize {} --nhid {} --dropouti {} --dropoute {}".format(lr, nhid, nhid, dropouti, dropoute).split(' ')
                print(command, filename)
                with open(filename, 'w') as f:
                    p = subprocess.run(command, stdout=f)
