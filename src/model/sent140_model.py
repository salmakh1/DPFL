import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.NLP_utils import get_word_emb_arr


class Base(nn.Module):
    """
    Base module containing the encoder and the recurrent module.
    """

    def __init__(self, emb, ninp, nhid, nlayers, dropout=0.5):
        super(Base, self).__init__()

        rnn_type = 'LSTM'
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                    options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)

        self.drop = nn.Dropout(dropout)
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, emb, hidden):
        self.rnn.flatten_parameters()
        # logging.info(f"the size of emb is {len(emb)} and of hidden is {len(hidden)}")
        output, hidden = self.rnn(emb, hidden)
        return output, hidden


class Classifier(nn.Module):
    """
    Classifier module containing the fully connected layers.
    """

    def __init__(self, nhid, ntoken):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(nhid, 10)
        self.decoder = nn.Linear(10, ntoken)

    def forward(self, output):
        output = F.relu(self.fc(output))
        decoded = self.decoder(output[-1, :, :])
        return decoded.t()


class RNNSent(nn.Module):
    """
    Container module with an encoder, a recurrent module, and a decoder.
    """

    def __init__(self, num_classes, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, emb_arr=None,
                 emb=None):
        super(RNNSent, self).__init__()
        # VOCAB_DIR = os.path.join(os.path.expanduser('~'), 'decentralised_learning/Federated_learning/src'
        #                                                           '/model/glove/embs.json')
        # logging.info(f"VOCAB_DIR is {VOCAB_DIR}")
        # emb, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
        self.encoder = torch.tensor(emb)
        self.base = Base(emb, ninp, nhid, nlayers, dropout)
        self.classifier = Classifier(nhid, ntoken)
        self.nhid = nhid
        self.rnn_type = 'LSTM'
        self.nlayers = nlayers
        # Tie weights if specified
        if tie_weights:
            if self.base.nhid != self.encoder.weight.size(1):
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.classifier.decoder.weight = self.encoder.weight

        self.device = torch.device('cuda')

    def forward(self, input, hidden):
        input = torch.transpose(input, 0, 1)
        batch_size = input.size(1)  # Get the batch size from the input tensor
        emb = torch.zeros((25, batch_size, 300))
        for i in range(25):
            for j in range(batch_size):
                emb[i, j, :] = self.encoder[input[i, j], :]
        emb = emb.to(self.device)
        emb = emb.view(300, batch_size, 25)
        if batch_size != hidden[0].size(1):
            hidden = (hidden[0][:, :batch_size, :], hidden[1][:, :batch_size, :])
        output, hidden = self.base(emb, hidden)
        decoded = self.classifier(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
