from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import numpy as np
import time


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim_en, hidden_dim_de, projected_size):
        super(AttentionLayer, self).__init__()
        self.linear1 = nn.Linear(hidden_dim_en, projected_size)
        self.linear2 = nn.Linear(hidden_dim_de, projected_size)
        self.linear3 = nn.Linear(projected_size, 1, False)

    def forward(self, out_e, h):
        '''
        out_e: batch_size * num_frames * en_hidden_dim
        h : batch_size * de_hidden_dim
        '''
        assert out_e.size(0) == h.size(0)
        batch_size, num_frames, _ = out_e.size()
        hidden_dim = h.size(1)

        h_att = h.unsqueeze(1).expand(batch_size, num_frames, hidden_dim)
        x = F.tanh(F.dropout(self.linear1(out_e)) + F.dropout(self.linear2(h_att)))
        x = F.dropout(self.linear3(x))
        a = F.softmax(x.squeeze(2))

        return a


def _smallest(matrix, k, only_first_row=False):
    if only_first_row:
        flatten = matrix[:1, :].flatten()
    else:
        flatten = matrix.flatten()
    args = np.argpartition(flatten, k)[:k]
    args = args[np.argsort(flatten[args])]
    return np.unravel_index(args, matrix.shape), flatten[args]


class luong_gate_attention(nn.Module):
    def __init__(self, hidden_size, emb_size, prob=0.1):
        super(luong_gate_attention, self).__init__()
        self.hidden_size, self.emb_size = hidden_size, emb_size
        self.linear_enc = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob),
                                        nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob))
        self.linear_in = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob),
                                       nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob))
        self.linear_out = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob),
                                        nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob))
        self.softmax = nn.Softmax(dim=-1)

    def init_context(self, context):
        self.context = context

    def forward(self, h, selfatt=False):
        if selfatt:
            gamma_enc = self.linear_enc(self.context)  # Batch_size * Length * Hidden_size
            gamma_h = gamma_enc.transpose(1, 2)  # Batch_size * Hidden_size * Length
            weights = torch.bmm(gamma_enc, gamma_h)  # Batch_size * Length * Length
            weights = self.softmax(weights / math.sqrt(512))
            c_t = torch.bmm(weights, gamma_enc)  # Batch_size * Length * Hidden_size
            output = self.linear_out(torch.cat([gamma_enc, c_t], 2)) + self.context
            output = output.transpose(0, 1)  # Length * Batch_size * Hidden_size
        else:
            gamma_h = self.linear_in(h).unsqueeze(0)
            # gamma_h = gamma_h.transpose(0, 1)
            weights = torch.bmm(self.context.unsqueeze(0), gamma_h).squeeze(2)  # batch*len
            weights = self.softmax(weights)
            c_t = torch.bmm(weights, self.context.unsqueeze(0)).squeeze(0).transpose(0,1
                                                                                     )
            output = self.linear_out(torch.cat([h, c_t], 1))

        return output, weights


class VisualEncoder(nn.Module):
    def __init__(self, opt):
        super(VisualEncoder, self).__init__()
        # embedding (input) layer options
        self.feat_size = opt.feat_size
        self.embed_dim = opt.word_embed_dim
        # rnn layer options
        self.rnn_type = opt.rnn_type
        self.num_layers = opt.num_layers
        self.hidden_dim = opt.hidden_dim
        self.dropout = opt.visual_dropout
        self.story_size = opt.story_size
        self.with_position = opt.with_position

        # visual embedding layer
        self.visual_emb = nn.Sequential(nn.Linear(self.feat_size, self.embed_dim),
                                        nn.BatchNorm1d(self.embed_dim),
                                        nn.ReLU(True))

        # visual rnn layer
        self.hin_dropout_layer = nn.Dropout(self.dropout)
        if self.rnn_type == 'gru' or self.rnn_type == 'gru_att' or self.rnn_type == 'bi_gru':
            self.rnn = nn.GRU(input_size=self.embed_dim, hidden_size=self.hidden_dim // 2,
                              dropout=self.dropout, batch_first=True, bidirectional=True)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_dim // 2,
                               dropout=self.dropout, batch_first=True, bidirectional=True)
        else:
            raise Exception("RNN type is not supported: {}".format(self.rnn_type))

        # residual part
        self.project_layer = nn.Linear(self.hidden_dim, self.embed_dim)
        self.relu = nn.ReLU()

        if self.with_position:
            self.position_embed = nn.Embedding(self.story_size, self.embed_dim)

    def init_hidden(self, batch_size, bi, dim):
        # the first parameter from the class
        weight = next(self.parameters()).data
        times = 2 if bi else 1
        if self.rnn_type == 'gru' or self.rnn_type == 'gru_att' or self.rnn_type == 'bi_gru':
            return Variable(weight.new(self.num_layers * times, batch_size, dim).zero_())
        else:
            return (Variable(weight.new(self.num_layers * times, batch_size, dim).zero_()),
                    Variable(weight.new(self.num_layers * times, batch_size, dim).zero_()))

    def forward(self, input, hidden=None):
        """
        inputs:
        - input  	(batch_size, 5, feat_size)
        - hidden 	(num_layers * num_dirs, batch_size, hidden_dim // 2)
        return:
        - out 		(batch_size, 5, rnn_size), serve as context
        """
        batch_size, seq_length = input.size(0), input.size(1)

        # visual embeded
        emb = self.visual_emb(input.view(-1, self.feat_size))
        emb = emb.view(batch_size, seq_length, -1)  # (Na, album_size, embedding_size)

        # visual rnn layer
        if hidden is None:
            hidden = self.init_hidden(batch_size, bi=True, dim=self.hidden_dim // 2)
        rnn_input = self.hin_dropout_layer(emb)  # apply dropout
        houts, hidden = self.rnn(rnn_input, hidden)

        # residual layer
        out = emb + self.project_layer(houts)
        out = self.relu(out)  # (batch_size, 5, embed_dim)

        if self.with_position:
            for i in xrange(self.story_size):
                position = Variable(input.data.new(batch_size).long().fill_(i))
                out[:, i, :] = out[:, i, :] + self.position_embed(position)

        return out, hidden
