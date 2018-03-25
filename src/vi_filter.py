"""
This is an attempt to do inference on more complex models using VI, including possibly applying particle filtering

There's HMM_GRU_MFVI, which is in the same vein as the other works

There's VRNN_MFVI and VRNN_PF, which do mean-field inference and particle filtering VI on a VRNN - via *sampling*
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.distributions import RelaxedOneHotCategorical, OneHotCategorical  # noqa: F401
import math
import pdb  # noqa: F401
import gc

from src.utils import print_in_epoch_summary, log_sum_exp, any_nans  # noqa: F401
from src.locked_dropout import LockedDropout  # noqa: F401
from src.embed_regularize import embedded_dropout  # noqa: F401
from src.hmm_filter import HMM_VI_Layers
from src.utils import show_memusage  # noqa: F401

LOG_2PI = math.log(2 * math.pi)


def check_allocation():
    for obj in gc.get_objects():
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            if obj.size()[0] == 20000:
                return True
    return False


class HMM_GRU_MFVI(HMM_VI_Layers):
    """
    This model isn't a VRNN - it's just a regular GRU language model with the additional information / inference from z
    being concatenated into the process
    """
    def __init__(self, z_dim, x_dim, hidden_size, nhid, word_dim, temp, temp_prior, params=None, *args, **kwargs):
        super(HMM_GRU_MFVI, self).__init__(z_dim, x_dim, hidden_size, nhid, word_dim, temp, temp_prior, params, *args, **kwargs)

        # set up the new generative model - we have to use the T, pi, and emit somehow
        # well it's emit and hidden
        # maybe have an LSTM over the hidden for the generation of x?
        # z is batch x hidden, self.hidden is hidden_size x z_dim
        # so x ~ Multi(self.emit @ torch.cat([z, h]))
        # and h = self.hidden_rnn(self.emit[true_x], h)
        self.hidden_rnn = nn.GRUCell(hidden_size, hidden_size)
        self.project = nn.Linear(hidden_size + z_dim, hidden_size)

    def init_inference(self, hmm_params):
        # generative model parameters
        self.T.data = hmm_params['T']
        self.pi.data = hmm_params['pi']
        self.emit.data = hmm_params['emit']

        self.project.bias.data.zero_()
        self.project.weight.data.zero_()
        self.project.weight.data[:, self.hidden_size:] = hmm_params['hidden']

        # inference network parameters
        self.inp_embedding.weight.data = hmm_params['inp_embedding.weight']

        self.encoder.weight_ih_l0.data = hmm_params['encoder.weight_ih_l0']
        self.encoder.weight_hh_l0.data = hmm_params['encoder.weight_hh_l0']
        self.encoder.bias_ih_l0.data = hmm_params['encoder.bias_ih_l0']
        self.encoder.bias_hh_l0.data = hmm_params['encoder.bias_hh_l0']
        self.encoder.weight_ih_l0_reverse.data = hmm_params['encoder.weight_ih_l0_reverse']
        self.encoder.weight_hh_l0_reverse.data = hmm_params['encoder.weight_hh_l0_reverse']
        self.encoder.bias_ih_l0_reverse.data = hmm_params['encoder.bias_ih_l0_reverse']
        self.encoder.bias_hh_l0_reverse.data = hmm_params['encoder.bias_hh_l0_reverse']

        self.logits.weight.data = hmm_params['logits.weight']
        self.logits.bias.data = hmm_params['logits.bias']

        # probably unnecessary
        self.enc = nn.ModuleList([self.inp_embedding, self.encoder])

    def forward(self, input, args, n_particles, test=False):
        """
        n_particles is interpreted as 1 for now to not screw anything up
        """
        n_particles = 1
        T = nn.Softmax(dim=0)(self.T)  # NOTE: not in log-space
        pi = nn.Softmax(dim=0)(self.pi)

        # run the input and teacher-forcing inputs through the embedding layers here
        seq_len, batch_sz = input.size()
        emb = self.inp_embedding(input)
        hidden = self.init_hidden(batch_sz, self.nhid, 2)  # bidirectional
        hidden_states, (_, _) = self.encoder(emb, hidden)
        hidden_states = hidden_states.repeat(1, n_particles, 1)

        # run the z-decoder at this point, evaluating the NLL at each step
        h = Variable(hidden_states.data.new(batch_sz, self.hidden_size).zero_())

        nlls = hidden_states.data.new(seq_len, batch_sz * n_particles)
        loss = 0

        # now a log-prob
        prior_probs = pi.unsqueeze(0).expand(batch_sz * n_particles, self.z_dim)

        logits = self.init_hidden(batch_sz * n_particles, self.z_dim, squeeze=True)
        feed = None

        for i in range(seq_len):
            # build the next z sample - not differentiable! we don't train the inference network
            logits = F.log_softmax(self.logits(hidden_states[i]), 1).detach()
            z = OneHotCategorical(logits=logits).sample()

            # this should be batch_sz x x_dim
            feed = self.project(torch.cat([h, z], 1))  # batch_sz x hidden_dim
            scores = torch.mm(feed, self.emit.t())  # batch_sz x x_dim

            NLL = nn.CrossEntropyLoss(reduce=False)(scores, input[i].repeat(n_particles))
            if NLL.data.mean() > 30:
                pdb.set_trace()
            KL = (logits.exp() * (logits - (prior_probs + 1e-16).log())).sum(1)
            loss += (NLL + KL)

            nlls[i] = NLL.data

            # set things up for next time
            if i != seq_len - 1:
                prior_probs = (T.unsqueeze(0) * z.unsqueeze(1)).sum(2)
                h = self.hidden_rnn(feed, h)  # feed the pseudo-choice into the RNN

        # now, we calculate the final log-marginal estimator
        return loss.sum(), nlls.sum(), (seq_len * batch_sz * n_particles), 0


class HMM_GRU_MFVI_Deep(HMM_GRU_MFVI):
    def __init__(self, *args, **kwargs):
        super(HMM_GRU_MFVI_Deep, self).__init__(*args, **kwargs)
        self.z_decoder = None
        self.logits = nn.Sequential(nn.Linear(2 * self.nhid, self.nhid),
                                    nn.ReLU(),
                                    nn.Linear(self.nhid, self.z_dim))

    def init_inference(self, hmm_params):
        # generative model parameters
        self.T.data = hmm_params['T']
        self.pi.data = hmm_params['pi']
        self.emit.data = hmm_params['emit']

        self.project.bias.data.zero_()
        self.project.weight.data.zero_()
        self.project.weight.data[:, self.hidden_size:] = hmm_params['hidden']

        # inference network parameters
        self.inp_embedding.weight.data = hmm_params['inp_embedding.weight']

        self.encoder.weight_ih_l0.data = hmm_params['encoder.weight_ih_l0']
        self.encoder.weight_hh_l0.data = hmm_params['encoder.weight_hh_l0']
        self.encoder.bias_ih_l0.data = hmm_params['encoder.bias_ih_l0']
        self.encoder.bias_hh_l0.data = hmm_params['encoder.bias_hh_l0']
        self.encoder.weight_ih_l0_reverse.data = hmm_params['encoder.weight_ih_l0_reverse']
        self.encoder.weight_hh_l0_reverse.data = hmm_params['encoder.weight_hh_l0_reverse']
        self.encoder.bias_ih_l0_reverse.data = hmm_params['encoder.bias_ih_l0_reverse']
        self.encoder.bias_hh_l0_reverse.data = hmm_params['encoder.bias_hh_l0_reverse']

        self.logits[0].weight.data = hmm_params['logits.0.weight']
        self.logits[0].bias.data = hmm_params['logits.0.bias']
        self.logits[2].weight.data = hmm_params['logits.2.weight']
        self.logits[2].bias.data = hmm_params['logits.2.bias']

        # probably unnecessary
        self.enc = nn.ModuleList([self.inp_embedding, self.encoder])
