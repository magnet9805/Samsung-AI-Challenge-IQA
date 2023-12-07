import dataset as d
from util.preprocessing import  *
import multiprocessing
import torch
import torch.nn as nn
import pandas as pd
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import random
import warnings
import dataset as d
from train.models.encoder_resnet import EncoderResnet
from train.models.decoder_seq import DecoderSeq
from train.models.seq2seq import Seq2seq
from torch import optim
import pandas as pd

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

CFG = {
        'IMG_SIZE':224,
        'EPOCHS':100, #Your Epochs,
        'LR':1e-5, #Your Learning Rate,
        'BATCH_SIZE': 8, #Your Batch Size,
        'SEED':41,
        'num_worker' : multiprocessing.cpu_count(),
        'EARLY_STOP' : 10
    }

all_data = pd.read_csv('./data/open/train.csv')
all_comments = ' '.join(all_data['comments']).split()
vocab = set(all_comments)
vocab = ['<PAD>', '<SOS>', '<EOS>'] + list(vocab)
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

test_data = pd.read_csv('./data/open/test.csv')
test_transform = d.ImageTransForm(CFG['IMG_SIZE'], 1, 1)
test_dataset = d.CustomDataset(test_data, 'test', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True,num_workers=CFG['num_worker'], pin_memory=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

hidden_dim = 256
embed_dim = 128
output_dim = len(vocab)
num_layers = 1

encoder = EncoderResnet(hidden_dim)
decoder = DecoderSeq(output_dim, embed_dim, hidden_dim, num_layers)
model = Seq2seq(encoder, decoder, device)

PATH = "seq2seq-model.pt"
model.load_state_dict(torch.load(PATH))
model.to(device)
model.eval()

predicted_mos_list = []
predicted_comments_list = []

def greedy_decode(model, image, max_length=50):
    image = image.unsqueeze(0).to(device)
    
    _, mos = model(image)
    
    output_sentence = []

    # 시작 토큰 설정
    current_token = torch.tensor([1]).to(device)
    hidden, _ = model.encoder.forward(image)
    hidden = hidden.unsqueeze(dim=0)

    for _ in range(50): # 50은 max_len
        out, hidden = model.decoder(current_token, hidden)

        current_token = out.argmax(dim=-1)

        # <EOS> 토큰에 도달하면 멈춤
        if current_token.item() == word2idx['<EOS>']:
            break

        # <SOS> 또는 <PAD> 토큰은 생성한 캡션에 추가하지 않음
        if current_token.item() not in [word2idx['<SOS>'], word2idx['<PAD>']]:
            output_sentence.append(idx2word[current_token.item()])

    return ' '.join(output_sentence), mos.item()

# 추론 과정
with torch.no_grad():
    for imgs, _, _ in tqdm(test_loader):
        for img in imgs:
            img = img.float().to(device)
            caption, mos = greedy_decode(model, img)
            predicted_mos_list.append(mos)
            predicted_comments_list.append(caption)

# 결과 저장
result_df = pd.DataFrame({
    'img_name': test_data['img_name'],
    'mos': predicted_mos_list,
    'comments': predicted_comments_list
})

# 예측 결과에 NaN이 있다면, 제출 시 오류가 발생하므로 후처리 진행
result_df['comments'] = result_df['comments'].fillna('Nice Image.')
result_df.to_csv('submit.csv', index=False)

print("Inference completed and results saved to submit.csv.")
