import torch
from tqdm import tqdm

def train(model, train_dataloader, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    mask = 0
    model.train()
    for img, mos, comment in tqdm(train_dataloader):
        x = img.to(device)
        y = mos.to(device)

        optimizer.zero_grad()
        out, mos_pred = model(x)
        loss = criterion(mos_pred.to(torch.float64), y.to(torch.float64))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        mask +=1
        if mask ==100:
            break

    return epoch_loss / len(train_dataloader)