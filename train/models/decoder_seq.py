import numpy as np
from common.time_layers import *
import torch
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