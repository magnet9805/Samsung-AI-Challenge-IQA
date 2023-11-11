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

cnn_backbone = models.resnet50(pretrained=True)
modules = list(cnn_backbone.children())[:-1]
print(modules)
# self.cnn = nn.Sequential(*modules)