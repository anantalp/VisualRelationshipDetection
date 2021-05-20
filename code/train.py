import os
import numpy as np
import torch
from tensorboardX import SummaryWriter

import configuration as cfg

# Function to train the model.
def training(run_id, epoch, data_loader, model, criterion, optimizer, writer):

    print('train at epoch {}'.format(epoch))

    # Check for GPU availability.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device being used is %s\n' % device)

    losses = []

    # model training.
    model.train()
    
    # CUDA initialization for GPU computations.
    model.cuda()    
    criterion.cuda()

    # iterate over data.
    for i, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device=device, dtype=torch.float)
        labels = labels.to(device)
        labels = labels.long()

        # zero the parameter gradients
        optimizer.zero_grad()

        # Computing the model
        outputs = model(inputs)

        # compute loss.
        loss = criterion(outputs, labels)

        # backward and optimizer in training phase.
        loss.backward()
        optimizer.step()

        # aggregate the loss.
        losses.append(loss.item())

        # reporting loss for every batch size.
        if i % cfg.batch_size == 0:
            print("Training Epoch ", epoch, "- Loss : ", loss.item())
        del loss

    # loading the model.
    save_dir = os.path.join(cfg.saved_models_dir, str(run_id))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save the model after every 5 epochs.
    if epoch % 5 == 0:
        save_file_path = os.path.join(save_dir, 'model_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)

    print('Training Epoch: %d, Loss: %.4f' % (epoch, np.mean(losses)))

    # tensorboard.
    writer.add_scalar('Training Loss', np.mean(losses), epoch)
    
    










