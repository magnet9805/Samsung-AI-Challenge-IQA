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

from train.trainer import trainer



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():

    CFG = {
        'IMG_SIZE':224,
        'EPOCHS':100, #Your Epochs,
        'LR':1e-5, #Your Learning Rate,
        'BATCH_SIZE': 32, #Your Batch Size,
        'SEED':41,
        'num_worker' : multiprocessing.cpu_count(),
        'EARLY_STOP' : 10
    }

    seed_everything(CFG['SEED']) # Seed 고정

    train_mean = (0.4194325, 0.3830166, 0.3490198)
    train_Std = (0.23905228, 0.2253936, 0.22334467)

    valid_mean = (0.4170096, 0.38036022, 0.34702352)
    valid_Std = (0.23896241, 0.22566794, 0.22329141)

    all_data = pd.read_csv('./data/open/train.csv')
    train_data = pd.read_csv('./data/open/train_data.csv')
    valid_data = pd.read_csv('./data/open/valid_data.csv')
    test_data = pd.read_csv('./data/open/test_data.csv')

    train_transform = d.ImageTransForm(CFG['IMG_SIZE'], train_mean, train_Std)
    valid_transform = d.ImageTransForm(CFG['IMG_SIZE'], valid_mean, valid_Std)

    train_dataset = d.CustomDataset(train_data, 'train', transform=train_transform)
    valid_dataset = d.CustomDataset(valid_data, 'valid', transform=valid_transform)



    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True,num_workers=CFG['num_worker'], pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=CFG['num_worker'], pin_memory=True)
    
    all_comments = ' '.join(all_data['comments']).split()
    vocab = set(all_comments)
    vocab = ['<PAD>', '<SOS>', '<EOS>'] + list(vocab)
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    hidden_dim = 256
    embed_dim = 128
    output_dim = len(vocab)
    num_layers = 1

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    encoder = EncoderResnet(hidden_dim)
    decoder = DecoderSeq(output_dim, embed_dim, hidden_dim, num_layers)
    model = Seq2seq(encoder, decoder, device)
    

    criterion_mos = nn.MSELoss()
    criterion_caption = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    criterion_mos.to(device)
    criterion_caption.to(device)
    encoder.to(device)
    decoder.to(device)
    model.to(device)
    
    dataloader_dict = {'train': train_loader, 'valid': valid_loader}
    criterion_dict = {'mos' : criterion_mos, 'caption': criterion_caption}

    train_history, valid_history = trainer(model, dataloader_dict, CFG['EPOCHS'], optimizer, criterion_dict, CFG['EARLY_STOP'],device,word2idx)
    return train_history, valid_history




if __name__ == "__main__":
    train_history, valid_history = main()

    pd = pd.DataFrame(columns=['train_loss', 'test_loss'],
                      data=[(train, valid) for train, valid in zip(train_history, valid_history)])
    pd.to_csv('loss.csv', encoding='cp949', index=False)
