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
        'EPOCHS':1000, #Your Epochs,
        'LR':1e-5, #Your Learning Rate,
        'BATCH_SIZE': 64, #Your Batch Size,
        'SEED':41,
        'num_worker' : multiprocessing.cpu_count(),
        'EARLY_STOP' : 10
    }

    seed_everything(CFG['SEED']) # Seed 고정

    train_mean = (0.42008194, 0.3838274, 0.34902292)
    train_Std = (0.23926373, 0.22593886, 0.22363442)

    test_mean = (0.4216005, 0.38125762, 0.34539804)
    test_Std = (0.23252015, 0.21890979, 0.21627444)




    train_data = pd.read_csv('./data/open/train.csv')
    test_data = pd.read_csv('./data/open/test.csv')
    train_transform = d.ImageTransForm(CFG['IMG_SIZE'], train_mean, train_Std)
    test_transform = d.ImageTransForm(CFG['IMG_SIZE'], test_mean, test_Std)

    train_dataset = d.CustomDataset(train_data, 'train', transform=train_transform)
    test_dataset = d.CustomDataset(test_data, 'test', transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=CFG['num_worker'], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=CFG['num_worker'], pin_memory=True)

    dataloader_dict = {'train' : train_loader, 'test' : test_loader}


    encoder = EncoderResnet(512)
    #torch.cuda.get_device_name(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.is_available()
    print(encoder)


    criterion = nn.MSELoss()
    optimizer = optim.Adam(encoder.parameters(), lr=1e-5)
    criterion.to(device)
    encoder.to(device)

    train_history, valid_history = trainer(encoder, dataloader_dict=dataloader_dict, criterion=criterion, num_epoch=CFG['EPOCHS'], optimizer=optimizer, device=device, early_stop=CFG['EARLY_STOP'])
    return train_history, valid_history


# # 단어 사전 생성
# all_comments = ' '.join(train_data['comments']).split()
# vocab = set(all_comments)
# vocab = ['<PAD>', '<SOS>', '<EOS>'] + list(vocab)
# word2idx = {word: idx for idx, word in enumerate(vocab)}
# idx2word = {idx: word for word, idx in word2idx.items()}
#
#
# # 모델, 손실함수, 옵티마이저
# model = encoder.EncoderCNN(len(vocab))
# criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'])
# optimizer = torch.optim.Adam(model.parameters(), lr=CFG['LR'])


# ## Train

# In[10]:


#
#
#
# # 학습
# model.train()
# for epoch in range(CFG['EPOCHS']):
#     total_loss = 0
#     loop = tqdm(train_loader, leave=True)
#     for imgs, comments in loop:
#         imgs = imgs.float()
#
#         # Batch Preprocessing
#         comments_tensor = torch.zeros((len(comments), len(max(comments, key=len)))).long()
#         for i, comment in enumerate(comments):
#             tokenized = ['<SOS>'] + comment.split() + ['<EOS>']
#             comments_tensor[i, :len(tokenized)] = torch.tensor([word2idx[word] for word in tokenized])
#             print(comments_tensor.size())
#
#         # Forward & Loss
#         predicted_comments = model(imgs, comments_tensor)
#         loss = criterion(predicted_comments.view(-1, len(vocab)), comments_tensor.view(-1))
#
#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#         loop.set_description(f"Epoch {epoch + 1}")
#         loop.set_postfix(loss=loss.item())
#
#     print(f"Epoch {epoch + 1} finished with average loss: {total_loss / len(train_loader):.4f}")
#
#
# # ## Inference & Submit
#
# # In[ ]:
#
#
# test_data = pd.read_csv('./data/open/test.csv')
# test_dataset = d.CustomDataset(test_data, transform)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#
# model.eval()
# predicted_mos_list = []
# predicted_comments_list = []
#
# def greedy_decode(model, image, max_length=50):
#     image = image.unsqueeze(0)
#     mos, _ = model(image)
#     output_sentence = []
#
#     # 시작 토큰 설정
#     current_token = torch.tensor([word2idx['<SOS>']])
#     hidden = None
#     features = model.cnn(image).view(image.size(0), -1)
#
#     for _ in range(max_length):
#         embeddings = model.embedding(current_token).unsqueeze(0)
#         combined = torch.cat([features.unsqueeze(1), embeddings], dim=2)
#         out, hidden = model.lstm(combined, hidden)
#
#         output = model.fc(out.squeeze(0))
#         _, current_token = torch.max(output, dim=1)
#
#         # <EOS> 토큰에 도달하면 멈춤
#         if current_token.item() == word2idx['<EOS>']:
#             break
#
#         # <SOS> 또는 <PAD> 토큰은 생성한 캡션에 추가하지 않음
#         if current_token.item() not in [word2idx['<SOS>'], word2idx['<PAD>']]:
#             output_sentence.append(idx2word[current_token.item()])
#
#     return mos.item(), ' '.join(output_sentence)
#
# # 추론 과정
# with torch.no_grad():
#     for imgs, _, _ in tqdm(test_loader):
#         for img in imgs:
#             img = img.float()
#             mos, caption = greedy_decode(model, img)
#             predicted_mos_list.append(mos)
#             predicted_comments_list.append(caption)
#
# # 결과 저장
# result_df = pd.DataFrame({
#     'img_name': test_data['img_name'],
#     'mos': predicted_mos_list,
#     'comments': predicted_comments_list  # 캡션 부분은 위에서 생성한 것을 사용
# })
#
# # 예측 결과에 NaN이 있다면, 제출 시 오류가 발생하므로 후처리 진행 (sample_submission.csv과 동일하게)
# result_df['comments'] = result_df['comments'].fillna('Nice Image.')
# result_df.to_csv('submit.csv', index=False)
#
# print("Inference completed and results saved to submit.csv.")
#
#
# # In[ ]:
#
#



if __name__ == "__main__":
    train_history, valid_history = main()

    pd = pd.DataFrame(columns=['train_loss', 'test_loss'],
                      data=[(train, valid) for train, valid in zip(train_history, valid_history)])
    pd.to_csv('loss.csv')
