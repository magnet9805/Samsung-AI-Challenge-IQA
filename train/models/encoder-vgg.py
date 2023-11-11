import torch
import torch.nn as nn
import torchvision.models as models



def get_model(models):
    return list(models.children())[:-1]

class VGG(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model_block =get_model(models.vgg16_bn())
        self.vgg = nn.Sequential(*model_block)
        
        
    def forward(self, x):
        return self.vgg(x)


vgg = VGG()
#
print(vgg)
# print(list(models.vgg16_bn().children())[:-1])