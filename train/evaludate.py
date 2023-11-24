import torch
from tqdm import tqdm


def evaluate(model, test_dataloader, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    mask = 0

    model.eval()
    import torch
    with torch.no_grad():
        for img, mos, comment in tqdm(test_dataloader):
            x = img.to(device)
            y = mos.to(device)
            out, mos_pred = model(x)
            loss = criterion(mos_pred.to(torch.float64), y.to(torch.float64))
            epoch_loss += loss.item()

    return epoch_loss / len(test_dataloader)