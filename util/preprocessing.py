import numpy as np
import torchvision
from PIL import Image, ImageFile


def get_channel_mean(data):
    meanR = np.mean([m[0] for m in data])
    meanG = np.mean([m[1] for m in data])
    meanB = np.mean([m[2] for m in data])

    return meanR, meanG, meanB

def get_channel_std(data):
    stdR = np.mean([m[0] for m in data])
    stdG = np.mean([m[1] for m in data])
    stdB = np.mean([m[2] for m in data])

    return stdR, stdG, stdB


def get_mea_std(path):
    meanRGB = []; stdRGB = []

    for index, path in enumerate(path):
        try :
            tensor = torchvision.transforms.ToTensor()
            img = Image.open(path).resize((224, 224)).convert("RGB")
            img =tensor(img)

            rgb = img.mean(dim=(1,2))
            meanRGB.append(rgb)

            std = img.std(dim=(1,2))
            stdRGB.append(std)

        except OSError as e:
            print(e)
            print(path)

    return meanRGB, stdRGB