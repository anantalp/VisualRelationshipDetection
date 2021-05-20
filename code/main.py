import sys, traceback, os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from datetime import datetime
from tensorboardX import SummaryWriter

from dataloader import OpenImageVRD
from models import define_model
from train_model import training
from evaluate_model import validation
import configuration as cfg


def combine(run_id, saved_models_dir, model_name, route):

    # Initialize tensorboard for the experiment in question by creating a folder using run_id.
    # save hyperparameters and certain input variables in tensorboard.
    writer = SummaryWriter(os.path.join(cfg.logs_dir, str(run_id)))
    writer.add_text("run_id", str(run_id), 0)
    writer.add_text("batch_size", str(cfg.batch_size), 0)
    writer.add_text("learning_rate", str(cfg.learning_rate), 0)
    writer.add_text("Data fraction", str(cfg.fraction), 0)
    writer.add_text('Model', str(model_name), 0)

    print("Preprocessing Data")
    print("run_id: ", run_id)
    print("Model name: ", model_name)
    print("Route: ", route)
    print("Batch Size: ", cfg.batch_size)
    print("Learning Rate: ", cfg.learning_rate)
    print("Dataset Fraction: ", cfg.fraction)

    # obtain data from train dataset.
    train_dataset = OpenImageVRD(cfg.root_dir, route, cfg.data_train_folder, cfg.data_train_csv, cfg.fraction)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    # obtain data from validation dataset.
    validation_dataset = OpenImageVRD(cfg.root_dir, route, cfg.data_val_folder, cfg.data_val_csv, cfg.fraction)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=cfg.batch_size, shuffle=True)

    # create if directory not found.
    # Use it to save the weight file.
    if not os.path.exists(saved_models_dir):
        os.makedirs(saved_models_dir)    

    # define the model
    model = define_model(model_name)
    
    # initialize optimizer, scheduler, and criterion.
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # loop for each epoch
    for epoch in range(cfg.num_epochs):
        try:
            training(run_id, epoch, train_dataloader,  model, criterion, optimizer, writer)
            validation(epoch, validation_dataloader, model, criterion, writer)
            scheduler.step()
        except:
            print("Epoch ", epoch, " failed")
            print('-'*30)
            traceback.print_exc(file=sys.stdout)
            print('-'*30)
            continue


if __name__ == "__main__":
    model = cfg.model[1]
    route = cfg.route[0]
    data_fraction = int(cfg.fraction*100)
    run_id = route+'_'+model+'_'+str(data_fraction)+'_'+datetime.today().strftime('%d-%m-%y_%H%M%S')
    combine(run_id, cfg.saved_models_dir, model, route)
    
    # pending: creating parser for command-line options and arguments.
