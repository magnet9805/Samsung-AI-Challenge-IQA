import time
from train.train import train
from train.evaludate import evaluate
import torch
from util.times import epoch_time
from torch.utils.tensorboard import SummaryWriter

def trainer(model, dataloader_dict, num_epoch, optimizer, criterion_dict, early_stop,device,word2idx):
    EPOCHS = num_epoch
    train_history, valid_history = [], []
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('./logs/')


    lowest_epoch = 0
    best_valid_loss = float('inf')
    for epoch in range(EPOCHS):
        start_time = time.monotonic()
        train_loss = train(model, dataloader_dict['train'], optimizer, criterion_dict, device, word2idx)
        valid_loss = evaluate(model, dataloader_dict['valid'], criterion_dict, device, word2idx)

        if valid_loss < best_valid_loss:
            lowest_epoch = epoch
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'seq2seq-model.pt')
        if early_stop > 0 and lowest_epoch + early_stop < epoch + 1:
            print("There is no improvement during last %d epochs." % early_stop)
            break
        train_history.append(train_loss)
        valid_history.append(valid_loss)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/valid", valid_loss, epoch)



        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Valid. Loss: {valid_loss:.3f}')
    writer.close()

    return train_history, valid_history