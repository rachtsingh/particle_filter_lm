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

from src.hmm import HMM_EM_Layers
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


def load_inference(self, hmm_params, deep=False):
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

    if deep:
        self.logits[0].weight.data = hmm_params['logits.0.weight']
        self.logits[0].bias.data = hmm_params['logits.0.bias']
        self.logits[2].weight.data = hmm_params['logits.2.weight']
        self.logits[2].bias.data = hmm_params['logits.2.bias']
    else:
        self.logits.weight.data = hmm_params['logits.weight']
        self.logits.bias.data = hmm_params['logits.bias']

    # probably unnecessary
    self.enc = nn.ModuleList([self.inp_embedding, self.encoder])


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
        self.hidden_rnn = nn.GRUCell(self.word_dim, hidden_size)
        self.project = nn.Linear(hidden_size + z_dim, hidden_size)

    def init_inference(self, hmm_params):
        load_inference(self, hmm_params)

    def forward(self, input, args, n_particles, test=False):
        """
        n_particles is interpreted as 1 for now to not screw anything up
        """
        if test:
            n_particles = 10
        else:
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
        h = Variable(hidden_states.data.new(batch_sz * n_particles, self.hidden_size).zero_())

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
            KL = (logits.exp() * (logits - (prior_probs + 1e-16).log())).sum(1)
            loss += (NLL + KL)

            nlls[i] = NLL.data

            # set things up for next time
            if i != seq_len - 1:
                prior_probs = (T.unsqueeze(0) * z.unsqueeze(1)).sum(2)
                h = self.hidden_rnn(emb[i].repeat(n_particles, 1), h)  # feed the next word into the RNN

        if n_particles != 1:
            loss = -log_sum_exp(-loss.view(n_particles, batch_sz), 0) + math.log(n_particles)
            NLL = -log_sum_exp(-nlls.view(seq_len, n_particles, batch_sz), 1) + math.log(n_particles)  # not quite accurate, but what can you do
        else:
            NLL = nlls

        # now, we calculate the final log-marginal estimator
        return loss.sum(), NLL.sum(), (seq_len * batch_sz), 0


class HMM_GRU_MFVI_Deep(HMM_GRU_MFVI):
    def __init__(self, *args, **kwargs):
        super(HMM_GRU_MFVI_Deep, self).__init__(*args, **kwargs)
        self.z_decoder = None
        self.logits = nn.Sequential(nn.Linear(2 * self.nhid, self.nhid),
                                    nn.ReLU(),
                                    nn.Linear(self.nhid, self.z_dim))

    def init_inference(self, hmm_params):
        load_inference(self, hmm_params, deep=True)


class HMM_GRU_Auto_Deep(HMM_GRU_MFVI_Deep):
    def __init__(self, *args, **kwargs):
        super(HMM_GRU_Auto_Deep, self).__init__(*args, **kwargs)
        self.z_decoder = nn.GRUCell(self.word_dim, 50)
        self.z_emb = nn.Linear(self.z_dim, 50)
        self.project_z = nn.Linear(50, self.z_dim)
        self.lockdrop = LockedDropout()
        self.dropout_x = 0.1

    def init_inference(self, hmm_params):
        load_inference(self, hmm_params, deep=True)

    def init_z_gru(self, params):
        gru, z_emb, project = params
        self.z_decoder = gru
        self.z_emb = z_emb
        self.project_z = project

    def forward(self, input, args, n_particles, test=False):
        """
        The major difference is that now we use a GRU to predict the prior z logits, instead of using a linear map
        T. I think trying to fit this GRU is really hard, I'm kind of concerned
        """
        if test:
            n_particles = 10
        else:
            n_particles = 1
        pi = F.log_softmax(self.pi, 0)

        # run the input and teacher-forcing inputs through the embedding layers here
        seq_len, batch_sz = input.size()
        emb = self.inp_embedding(input)
        hidden = self.init_hidden(batch_sz, self.nhid, 2)  # bidirectional
        hidden_states, (_, _) = self.encoder(emb, hidden)
        hidden_states = hidden_states.repeat(1, n_particles, 1)

        # run the z-decoder at this point, evaluating the NLL at each step
        h = Variable(hidden_states.data.new(batch_sz * n_particles, self.hidden_size).zero_())

        nlls = hidden_states.data.new(seq_len, batch_sz * n_particles)
        loss = 0

        # now a log-prob
        prior_logits = pi.unsqueeze(0).expand(batch_sz * n_particles, self.z_dim)
        prior_h = Variable(torch.zeros(batch_sz * n_particles, 50).cuda())

        logits = self.init_hidden(batch_sz * n_particles, self.z_dim, squeeze=True)
        feed = None

        # use dropout on the teacher-forcing
        x_emb = self.lockdrop(emb, self.dropout_x)

        for i in range(seq_len):
            # build the next z sample - not differentiable! we don't train the inference network
            logits = F.log_softmax(self.logits(hidden_states[i]), 1).detach()
            z = OneHotCategorical(logits=logits).sample()

            # this should be batch_sz x x_dim
            scores = torch.mm(self.project(torch.cat([h, z], 1)), self.emit.t())

            NLL = nn.CrossEntropyLoss(reduce=False)(scores, input[i].repeat(n_particles))
            KL = (logits.exp() * (logits - prior_logits)).sum(1)
            loss += (NLL + KL)

            nlls[i] = NLL.data

            # set things up for next time
            if i != seq_len - 1:
                feed = torch.cat([emb[i].repeat(n_particles, 1), self.z_emb(z)], 1)
                prior_h = self.z_decoder(feed, prior_h)
                prior_logits = F.log_softmax(self.project_z(prior_h), 1)
                h = self.hidden_rnn(x_emb[i].repeat(n_particles, 1), h)  # feed the next word into the RNN

        if n_particles != 1:
            loss = -log_sum_exp(-loss.view(n_particles, batch_sz), 0) + math.log(n_particles)
            NLL = -log_sum_exp(-nlls.view(seq_len, n_particles, batch_sz), 1) + math.log(n_particles)  # not quite accurate, but what can you do
        else:
            NLL = nlls

        # now, we calculate the final log-marginal estimator
        return loss.sum(), NLL.sum(), (seq_len * batch_sz), 0


class HMM_LSTM_Auto_Deep(HMM_GRU_MFVI_Deep):
    """
    TODO: make 50 an arg (it's Z_EMB and Z_HID)
    """
    def __init__(self, *args, **kwargs):
        super(HMM_LSTM_Auto_Deep, self).__init__(*args, **kwargs)
        Z_EMB = 50
        Z_HID = 50
        self.hidden_rnn = nn.LSTMCell(self.word_dim, self.hidden_size)
        self.z_decoder = nn.LSTMCell(self.word_dim + Z_EMB, Z_HID)
        self.z_emb = nn.Linear(self.z_dim, Z_EMB)
        self.project_z = nn.Linear(Z_HID, self.z_dim)
        self.lockdrop = LockedDropout()
        self.dropout_x = 0.3

    def init_inference(self, hmm_params):
        load_inference(self, hmm_params, deep=True)

    def init_z_gru(self, params):
        lstm, z_emb, project = params
        self.z_decoder = lstm
        self.z_emb = z_emb
        self.project_z = project

    def forward(self, input, args, n_particles, test=False):
        """
        s/GRU/LSTM/g
        """
        if test:
            n_particles = 10
        else:
            n_particles = 1
        pi = F.log_softmax(self.pi, 0)

        # run the input and teacher-forcing inputs through the embedding layers here
        seq_len, batch_sz = input.size()
        emb = self.inp_embedding(input)
        hidden = self.init_hidden(batch_sz, self.nhid, 2)  # bidirectional
        hidden_states, (_, _) = self.encoder(emb, hidden)
        hidden_states = hidden_states.repeat(1, n_particles, 1)

        # run the z-decoder at this point, evaluating the NLL at each step
        h = (Variable(hidden_states.data.new(batch_sz * n_particles, self.hidden_size).zero_()),
             Variable(hidden_states.data.new(batch_sz * n_particles, self.hidden_size).zero_()))

        nlls = hidden_states.data.new(seq_len, batch_sz * n_particles)
        loss = 0

        # now a log-prob
        prior_logits = pi.unsqueeze(0).expand(batch_sz * n_particles, self.z_dim)
        prior_h = (Variable(torch.zeros(batch_sz * n_particles, 50).cuda()),
                   Variable(torch.zeros(batch_sz * n_particles, 50).cuda()))

        logits = self.init_hidden(batch_sz * n_particles, self.z_dim, squeeze=True)
        feed = None

        x_emb = self.lockdrop(emb, self.dropout_x)

        for i in range(seq_len):
            # build the next z sample - not differentiable! we don't train the inference network
            logits = F.log_softmax(self.logits(hidden_states[i]), 1).detach()
            z = OneHotCategorical(logits=logits).sample()

            # this should be batch_sz x x_dim
            scores = torch.mm(self.project(torch.cat([h[0], z], 1)), self.emit.t())

            NLL = nn.CrossEntropyLoss(reduce=False)(scores, input[i].repeat(n_particles))
            KL = (logits.exp() * (logits - prior_logits)).sum(1)
            loss += (NLL + KL)

            nlls[i] = NLL.data

            # set things up for next time
            if i != seq_len - 1:
                feed = torch.cat([emb[i].repeat(n_particles, 1), self.z_emb(z)], 1)
                prior_h = self.z_decoder(feed, prior_h)
                prior_logits = F.log_softmax(self.project_z(prior_h[0]), 1)
                h = self.hidden_rnn(x_emb[i].repeat(n_particles, 1), h)  # feed the next word into the RNN

        if n_particles != 1:
            loss = -log_sum_exp(-loss.view(n_particles, batch_sz), 0) + math.log(n_particles)
            NLL = -log_sum_exp(-nlls.view(seq_len, n_particles, batch_sz), 1) + math.log(n_particles)  # not quite accurate, but what can you do
        else:
            NLL = nlls

        # now, we calculate the final log-marginal estimator
        return loss.sum(), NLL.sum(), (seq_len * batch_sz), 0


class VRNN_LSTM_Auto_Concrete(HMM_LSTM_Auto_Deep):
    """
    The big difference with the parent is that this model attempts to learn
    the inference network directly via the Concrete distribution.
    """
    def forward(self, input, args, n_particles, test=False):
        if test:
            n_particles = 10
        else:
            n_particles = 1
        pi = F.log_softmax(self.pi, 0)

        temp = Variable(self.pi.data.new([args.temp]))
        temp_prior = Variable(self.pi.data.new([args.temp_prior]))

        # run the input and teacher-forcing inputs through the embedding layers here
        seq_len, batch_sz = input.size()
        emb = self.inp_embedding(input)
        hidden = self.init_hidden(batch_sz, self.nhid, 2)  # bidirectional
        hidden_states, (_, _) = self.encoder(emb, hidden)
        hidden_states = hidden_states.repeat(1, n_particles, 1)

        # run the z-decoder at this point, evaluating the NLL at each step
        h = (Variable(hidden_states.data.new(batch_sz * n_particles, self.hidden_size).zero_()),
             Variable(hidden_states.data.new(batch_sz * n_particles, self.hidden_size).zero_()))

        nlls = hidden_states.data.new(seq_len, batch_sz * n_particles)
        loss = 0

        # now a log-prob
        prior_logits = pi.unsqueeze(0).expand(batch_sz * n_particles, self.z_dim)
        prior_h = (Variable(torch.zeros(batch_sz * n_particles, 50).cuda()),
                   Variable(torch.zeros(batch_sz * n_particles, 50).cuda()))

        logits = self.init_hidden(batch_sz * n_particles, self.z_dim, squeeze=True)
        feed = None

        x_emb = self.lockdrop(emb, self.dropout_x)

        for i in range(seq_len):
            # build the next z sample - not differentiable! we don't train the inference network
            logits = F.log_softmax(self.logits(hidden_states[i]), 1).detach()

            if test:
                q = OneHotCategorical(logits=logits)
                p = OneHotCategorical(logits=prior_logits)
                z = q.sample()
            else:
                q = RelaxedOneHotCategorical(temperature=temp, logits=logits)
                p = RelaxedOneHotCategorical(temperature=temp_prior, logits=prior_logits)
                z = q.rsample()

            # this should be batch_sz x x_dim
            scores = torch.mm(self.project(torch.cat([h[0], z], 1)), self.emit.t())

            NLL = nn.CrossEntropyLoss(reduce=False)(scores, input[i].repeat(n_particles))
            KL = q.log_prob(z) - p.log_prob(z)
            if test:
                loss += (NLL + KL)
            else:
                loss += (NLL + args.anneal * KL)

            nlls[i] = NLL.data

            # set things up for next time
            if i != seq_len - 1:
                feed = torch.cat([emb[i].repeat(n_particles, 1), self.z_emb(z)], 1)
                prior_h = self.z_decoder(feed, prior_h)
                prior_logits = F.log_softmax(self.project_z(prior_h[0]), 1)
                h = self.hidden_rnn(x_emb[i].repeat(n_particles, 1), h)  # feed the next word into the RNN

        if n_particles != 1:
            loss = -log_sum_exp(-loss.view(n_particles, batch_sz), 0) + math.log(n_particles)
            NLL = -log_sum_exp(-nlls.view(seq_len, n_particles, batch_sz), 1) + math.log(n_particles)  # not quite accurate, but what can you do
        else:
            NLL = nlls

        # now, we calculate the final log-marginal estimator
        return loss.sum(), NLL.sum(), (seq_len * batch_sz), 0


class HMM_Joint_LSTM(HMM_EM_Layers):
    """
    This is a reimplementation of the joint-hybrid work of (Krakovna, 2016)
    """
    def __init__(self, z_dim, x_dim, hidden_size, word_dim, lstm_hidden_size, separate_opt=False):
        super(HMM_Joint_LSTM, self).__init__(z_dim, x_dim, hidden_size)
        self.inp_embedding = nn.Embedding(x_dim, word_dim)
        self.lstm = nn.LSTMCell(word_dim, lstm_hidden_size)
        self.project = nn.Linear(lstm_hidden_size + z_dim, x_dim)
        self.separate_opt = separate_opt
        self.lstm_hidden_size = lstm_hidden_size

    def forward(self, input, args, test=False):
        seq_len, batch_size = input.size()
        # compute the loss as the sum of the forward-backward loss
        alpha, _, log_marginal = self.forward_backward(input)
        emb = self.inp_embedding(input)
        T = F.log_softmax(self.T, 0)
        pi = F.log_softmax(self.pi, 0).unsqueeze(0).expand(batch_size, self.z_dim)
        if self.separate_opt:
            pi = pi.detach()
            T = T.detach()

        h = (Variable(torch.zeros(batch_size, self.lstm_hidden_size).cuda()),
             Variable(torch.zeros(batch_size, self.lstm_hidden_size).cuda()))

        NLL = 0

        # now, compute the filtered posterior and together with the LSTM feed data into the net-output
        # note that \alpha(t) contains information about the current x, so we need to prop forward
        current_state = None
        for i in range(seq_len):
            if i == 0:
                hmm_post = pi
            else:
                hmm_post = log_sum_exp(T.unsqueeze(0) + current_state.unsqueeze(1), 2)

            scores = self.project(torch.cat([h[0], hmm_post], 1))
            NLL += nn.CrossEntropyLoss(size_average=False)(scores, input[i])

            # feed information from the current state into the next prediction (i.e. teacher-forcing)
            h = self.lstm(emb[i], h)
            current_state = F.log_softmax(alpha[i], 1)
            if self.separate_opt:
                current_state = current_state.detach()

        return (-log_marginal.sum() + NLL.sum()), NLL.data.sum()

    def evaluate(self, data_source, args, num_samples=None):
        self.eval()
        total_loss = 0
        total_nll = 0
        total_tokens = 0

        for batch in data_source:
            if args.cuda:
                batch = batch.cuda()
            data = Variable(batch.squeeze(0).t().contiguous())  # squeeze for 1 billion
            total_tokens += (data.size()[0] * data.size()[1])
            loss, nll = self.forward(data, args, test=True)
            loss = loss.sum()
            total_loss += loss.detach().data
            total_nll += nll

        if args.dataset != '1billion':
            total_tokens = 1

        return total_nll / float(total_tokens), total_loss[0] / float(total_tokens), (total_loss[0] - total_nll)/float(total_tokens)

    def train_epoch(self, train_data, optimizer, epoch, args, num_samples=None):
        self.train()
        total_loss = 0

        for i, batch in enumerate(train_data):
            if args.cuda:
                batch = batch.cuda()
            data = Variable(batch.squeeze(0).t().contiguous())  # squeeze for 1 billion
            optimizer.zero_grad()
            loss, nll = self.forward(data, args, test=False)
            tokens = (data.size()[0] * data.size()[1])
            loss = loss.sum() / tokens
            loss.backward()
            optimizer.step()
            total_loss += loss.detach()
            if (i + 1) % args.log_interval == 0:
                print("total loss: {:.3f}, nll: {:.3f}".format(loss.data[0], nll / tokens))

        # actually ignored
        return total_loss.data[0]
