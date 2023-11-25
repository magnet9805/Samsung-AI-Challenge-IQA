import time
from train.train import train
from train.evaludate import evaluate
import torch
from util.times import epoch_time
def trainer(model, dataloader_dict, num_epoch, optimizer, criterion, early_stop,device):
    EPOCHS = num_epoch
    train_history, valid_history = [], []

    lowest_epoch = 0
    best_valid_loss = float('inf')
    for epoch in range(EPOCHS):
        start_time = time.monotonic()
        train_loss = train(model, dataloader_dict['train'], optimizer, criterion, device)
        valid_loss = evaluate(model, dataloader_dict['valid'], criterion, device)

        if valid_loss < best_valid_loss:
            lowest_epoch = epoch
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'encoder-model.pt')
        if early_stop > 0 and lowest_epoch + early_stop < epoch + 1:
            print("There is no improvement during last %d epochs." % early_stop)
            break
        train_history.append(train_loss)
        valid_history.append(valid_loss)


        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Valid. Loss: {valid_loss:.3f}')
    
    return train_history, valid_history