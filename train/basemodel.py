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

class BaseModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super(BaseModel, self).__init__()

        # Image feature extraction using ResNet50
        self.cnn_backbone = models.resnet50(pretrained=True)
        # Remove the last fully connected layer to get features
        modules = list(self.cnn_backbone.children())[:-1]
        self.cnn = nn.Sequential(*modules)

        # Image quality assessment head
        self.regression_head = nn.Linear(2048, 1)  # ResNet50 last layer has 2048 features

        # Captioning head
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim + 2048, hidden_dim)  # Image features and caption embeddings as input
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, captions=None):
        # CNN
        features = self.cnn(images)
        features_flat = features.view(features.size(0), -1)

        # Image quality regression
        mos = self.regression_head(features_flat)

        # LSTM captioning
        if captions is not None:
            embeddings = self.embedding(captions)
            # Concatenate image features and embeddings for each word in the captions
            combined = torch.cat([features_flat.unsqueeze(1).repeat(1, embeddings.size(1), 1), embeddings], dim=2)
            lstm_out, _ = self.lstm(combined)
            outputs = self.fc(lstm_out)
            return mos, outputs
        else:
            return mos, None