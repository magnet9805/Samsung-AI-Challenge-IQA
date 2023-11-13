# class Seq2seq(BaseModel):
#     def __init__(self, vocab_size, wordvec_size, hidden_size):
#         V, D, H = vocab_size, wordvec_size, hidden_size
#         self.decoder = Decoder(V, D, H)
#         self.softmax = TimeSoftmaxWithLoss()

#         self.params = self.decoder.params
#         self.grads = self.decoder.grads

#     def forward(self, imgfeature, ts):
#         decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]

#         h = imgfeature
#         score = self.decoder.forward(decoder_xs, h)
#         loss = self.softmax.forward(score, decoder_ts)
#         return loss

#     def backward(self, dout=1):
#         dout = self.softmax.backward(dout)
#         dout = self.decoder.backward(dout)
#         return dout

#     def generate(self, imgfeature, start_id, sample_size):
#         h = imgfeature
#         sampled = self.decoder.generate(h, start_id, sample_size)
#         return sampled