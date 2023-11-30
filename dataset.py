import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

path = os.path.dirname(os.path.abspath(__file__))


class ImageTransForm():
    def __init__(self, resize, mean, std):
        self.data_tranform = {
            'train' : transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'valid' : transforms.Compose(
                [
                    transforms.Resize(224),                            
                    transforms.CenterCrop(resize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ]
            )
        }

    def __call__(self, img, phase):
        return self.data_tranform[phase](img)
    

class CustomDataset(Dataset):
    def __init__(self, dataframe, phase, transform=None ):
        dataframe[['path', 'extension']] = dataframe['img_path'].str.split(".", expand=True)[[1, 2]]
        dataframe = dataframe.drop('img_path', axis=1)
        self.dataframe = dataframe
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path : str= f"{path}/data/open{self.dataframe.iloc[idx]['path']}.{self.dataframe.iloc[idx]['extension']}"

        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img, self.phase)

        # mos column 존재 여부에 따라 값을 설정
        mos = float(self.dataframe.iloc[idx]['mos']) if 'mos' in self.dataframe.columns else 0.0
        comment = self.dataframe.iloc[idx]['comments'] if 'comments' in self.dataframe.columns else ""

        return img, mos, comment
    
