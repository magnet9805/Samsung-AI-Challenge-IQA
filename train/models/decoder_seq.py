import numpy as np
from common.time_layers import *
import torch
import torch.nn as nn

class DecoderSeq:
    def __init__(self, output_dim, embed_size, hidden_size, num_layers ):
        , H = vocab_size, hidden_size

        self.embed = nn.Embedding(V, H)

        # self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.lstm = nn.LSTM(H, H, 1)

        # self.affine = TimeAffine(affine_W, affine_b)
        self.affine = nn.Linear(H, D)
        

        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, imgfeature):
        self.lstm.set_state(imgfeature)

        out = self.embed.forward(xs)
        out = self.lstm.forward(out)
        score = self.affine.forward(out)
        return score