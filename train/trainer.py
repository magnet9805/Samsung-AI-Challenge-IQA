import time
from train.train import train
from train.evaludate import evaluate
import torch
from util.times import epoch_time
def trainer(model, dataloader_dict, num_epoch, optimizer, criterion, device):
    EPOCHS = num_epoch

    best_valid_loss = float('inf')
    for epoch in range(EPOCHS):
        start_time = time.monotonic()
        train_loss, train_acc = train(model, dataloader_dict['train'], optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, dataloader_dict['test'], criterion, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'encoder-model.pt')

        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Valid. Loss: {valid_loss:.3f} |  Valid. Acc: {valid_acc * 100:.2f}%')