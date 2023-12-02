import torch.nn as nn
import torch
import random


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, imgs, comments, teacher_forcing_ratio=0.5):
        batch_size = comments.shape[0]
        seq_len = comments.shape[1]
        vocab_size = self.decoder.output_dim
        
        input = comments[:, 0].to(self.device)
        target = comments[:, 1:].to(self.device)

        outputs = torch.zeros(batch_size, seq_len - 1, vocab_size).to(self.device)

        # imgfeature 뽑기 (디코더의 첫번째 h) 및 Encoder mos 저장
        decoder_hidden, mos = self.encoder.forward(imgs)

        # dimension 맞춰주기
        decoder_hidden = decoder_hidden.unsqueeze(dim=0)  # (1, batch_size, hidden_size)

        for t in range(seq_len - 1):
            decoder_out, decoder_hidden = self.decoder(input, decoder_hidden)
            outputs[:, t, :] = decoder_out.squeeze(1)

            # 확률 가장 높게 예측한 토큰
            topi = decoder_out.argmax(dim=-1)

            # 50% 확률로 다음 토큰값으로 예측값 or 실제값
            teacher_force = random.random() < teacher_forcing_ratio
            input = target[:, t] if teacher_force else topi

        # outputs = outputs.argmax(dim=-1)
        return outputs, mos  # (seq_len, batch_size, output_size)




