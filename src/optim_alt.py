import torch
from torch.optim import Optimizer
import itertools


class MixedOpt(Optimizer):
    def __init__(self, inf_params, gen_params, inf_lr, gen_lr):
        self.inf_opt = torch.optim.Adam(inf_params, inf_lr)
        self.gen_opt = torch.optim.SGD(gen_params, gen_lr)

    @property
    def param_groups(self):
        return itertools.chain(self.inf_opt.param_groups, self.gen_opt.param_groups)

    def zero_grad(self):
        self.inf_opt.zero_grad()
        self.gen_opt.zero_grad()

    def step(self):
        self.inf_opt.step()
        self.gen_opt.step()
