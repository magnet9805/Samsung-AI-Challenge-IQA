import torch.nn as nn

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device


    def forward(self, xs, ts): # xs = 이미지, ts = 문장
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]

        h, _ = self.encoder.forward(xs)
        score = self.decoder.forward(decoder_xs, h)
        loss = self.softmax.forward(score, decoder_ts)
        return loss

