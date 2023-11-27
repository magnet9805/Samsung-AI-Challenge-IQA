import numpy as np
from common.time_layers import *
import torch
<<<<<<< HEAD
from torch import nn

class DecoderSeq:
    def __init__(self, output_dim, embed_size, hidden_size, num_layers):
        O, E, H, N = output_dim, embed_size, hidden_size, num_layers

        self.embed = nn.Embedding(O, E)
        self.lstm = nn.LSTM(E, H, N, batch_first=True)
        self.affine = nn.Linear(H, O)

    def forward(self, input, hidden, cell):

        out = self.embed.forward(input)
        out, (hidden, cell) = self.lstm.forward(out, (hidden, cell))
        prediction = self.affine.forward(out)
        return prediction, hidden, cell
=======
import torch.nn as nn

class DecoderSeq(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers):
        super().__init__()

        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embed = nn.Embedding(output_dim, self.embed_dim)
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.affine = nn.Linear(self.hidden_dim, output_dim)
        #### self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden): # input : comments tensor / hidden : 이미지 feature

        # input = [batch_size]

        #input = [1, batch_size]
        input = input.unsqueeze(0)

        # out = [1, batch_size, embed_dim]
        out = self.embed(input)

        # out = [batch_size, seq_len, hidden_dim]
        out, hidden, _ = self.lstm(out, hidden)

        #### out = self.softmax(self.affine(out[0]))

        # out = [batch_size, output_dim]
        out = self.affine(out.squeeze(1))
        return out, hidden
>>>>>>> bdc553db1447fb56c0c00b02b576ca48ff746855
