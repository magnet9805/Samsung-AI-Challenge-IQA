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
warnings.filterwarnings(action='ignore') 

class EncoderCNN(nn.Module):
    def __init__(self, embed_size=256):
        super(EncoderCNN, self).__init__()

        # Image feature extraction using ResNet50
        self.cnn_backbone = models.resnet50(pretrained=True)
        # Remove the last fully connected layer to get features
        modules = list(self.cnn_backbone.children())[:-1]
        self.cnn = nn.Sequential(*modules)

        # 마지막 블록을 제거한 resnet에서 뽑아낸 이미지 feature를 FC layer를 붙여 embed_size vector가 출력 되도록 함
        self.linear = nn.Linear(self.cnn.fc.in_features, embed_size)

        # LSTM에 넣어줄 input은 vector 형태이므로 1D Batch normalization 수행
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        # CNN
        features = self.cnn(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))

        return features