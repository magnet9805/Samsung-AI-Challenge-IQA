import numpy as np

def get_mean(data):
    meanRGB = []; stdRGB = []
    for x, _ in data:
        mean = np.mean(x.numpy(), axis=(1, 2))
        meanRGB.append(mean)

        std = np.std(x.numpy(), axis=(1, 2))
        stdRGB.append(std)
    return  meanRGB, stdRGB


def get_channel_mean(data):
    meanR = np.mean([m[0] for m in data])
    meanG = np.mean([m[1] for m in data])
    meanB = np.mean([m[2] for m in data])

    return meanR, meanG, meanB

def get_channel_std(data):
    stdR = np.std([m[0] for m in data])
    stdG = np.std([m[1] for m in data])
    stdB = np.std([m[2] for m in data])

    return stdR, stdG, stdB