from train.models.decoder_seq import DecoderSeq
from common.time_layers import *
from common.base_model import *
from train.models.encoder_resnet import EncoderResnet

class Seq2seq(BaseModel):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        self.encoder = EncoderResnet(D)
        self.decoder = DecoderSeq(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.decoder.params
        self.grads = self.decoder.grads

    def forward(self, xs, ts): # xs = 이미지, ts = 문장
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]

        h, _ = self.encoder.forward(xs)
        score = self.decoder.forward(decoder_xs, h)
        loss = self.softmax.forward(score, decoder_ts)
        return loss

    def backward(self, dout=1):
        dout = self.softmax.backward(dout)
        dout = self.decoder.backward(dout)
        return dout

    def generate(self, xs, start_id, sample_size): # xs = 이미지
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled