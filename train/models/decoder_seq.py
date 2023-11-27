import numpy as np
import torch.nn as nn

class DecoderSeq(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers):
        super().__init__()
        
        # output_dim == vocab_size
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embed = nn.Embedding(output_dim, self.embed_dim)
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.affine = nn.Linear(self.hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden): # input : comments tensor / hidden : 이미지 feature

        # input = [batch_size]

        #input = [1, batch_size]
        input = input.view(1,-1)

        # out = [1, batch_size, embed_dim]
        out = self.embed(input)

        # out = [batch_size, seq_len, hidden_dim]
        # hidden = [num_layers, batch_size, hidden_size]
        out, hidden, _ = self.lstm(out, hidden)

        # out = [batch_size, output_dim]
        prediction = self.softmax(self.affine(out[0]))
        return prediction, hidden

