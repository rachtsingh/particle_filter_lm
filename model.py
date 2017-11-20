"""
This is the baseline model, which uses a RNN-LM directly.
"""
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import repackage_hidden
import numpy as np
import pdb

from locked_dropout import LockedDropout

class SequentialHidden(nn.Module):
    """
    This module computes / samples from q(z_t | z_{1: t - 1}, x), where everything is
    essentially a K-dimensional Gaussian for now.

    inputs:
    - input_sentence (RNN-encoded [bsz x ninp x L], where L is the maximum size in the batch)
    - z_previous ([bsz x K])

    internally: stores
    
    returns:
    - z_next
    """

class PFLM(nn.Module):
    """
    Full container that is essentially API-equivalent to the RNNModel, and contains inside it
    all the things we need
    """

    def __init__(self, ntoken, ninp, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False, bidirectional=False):
        super(PFLM, self).__init__()
        self.lockdrop = LockedDropout

        # apparently ninp and nhid are the same
        self.input_embedding = nn.Embedding(ntoken, ninp)
        self.output_embedding = nn.Linear(nhid, ntoken) 
        if tie_weights:
            self.decoder.weight = self.encoder.weight
        
        self.encoders = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0, bidirectional=bidirectional) for l in range(nlayers)]
        self.encoders = torch.nn.ModuleList(self.encoders)
    
        # initialize the embeddings
        self.init_weights()
        
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        """
        This fully processes the input, which is [batch_size x max_sent_len]
        
        This is the templated mode, where we generate the zs, then generate the ys
        (we probably want the other thing)
        """
        

    def elbo(self, zs, generation, target):
        criterion = torch.nn.CrossEntropyLoss(size_average=False)
        NLL = criterion(generation, target)

        
