import torch.nn as nn
import torchvision.models as models
class EncoderResnet(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.cnn_backbone = models.resnet50(pretrained=True)

        modules = list(self.cnn_backbone.children())[:-1]
        self.cnn = nn.Sequential(*modules)


        self.fc1 = nn.Linear(2048, 1024)
        self.fc_drop_out = nn.Dropout()

        self.out = nn.Linear(1024, output-1)

        self.mos = nn.Linear(1024, 1)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, img):
        features = self.cnn(img)
        features = features.view(features.size(0), -1)
        feature_out = self.relu(self.fc1(features))

        mos = self.mos(feature_out)
        out = self.out(feature_out)

        return out, mos

# model = EncoderResnet(512)

# from torchsummary import summary
# print(summary(model, input_size=(3, 224,224), batch_size=1))



