import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tensorboardX import SummaryWriter

from models import define_model
from dataloader import OpenImageVRD
import configuration as cfg


# Function that validates a model.
def validation(epoch, data_loader, model, criterion , writer):

    print('validation at epoch {}'.format(epoch))

    # Check for GPU availability.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device being used is %s\n' % device)

    # model evaluation.
    model.eval()
    
    losses = []
    predictions, ground_truth = [], []
    
    # Iterate over data.
    for i, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device=device, dtype=torch.float)
        labels1 = np.array(labels, dtype='long')
        labels = labels.to(device)
        labels = labels.long()

        # Appending groundtruths for each index of the dataloader.
        ground_truth.extend(labels1)

        # Computing the model.
        outputs = model(inputs)
        
        # compute loss and its aggregate.
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        # Final output of our model.
        outputs = torch.nn.Softmax()(outputs)
        outputs = torch.argmax(outputs, dim=1)
        
        # Appending the predictions for each index of the dataloader.
        predictions.extend(outputs.cpu().data.numpy())

        # reporting loss for every batch size.
        if i % cfg.batch_size == 0:
            print("Validation Epoch ", epoch, "- Loss : ", loss.item())
        del loss
        del outputs, labels

    # creating an array of groundtruth and predictions to compute classification metric.
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
  
    # Report classification metrics.
    results_actions = precision_recall_fscore_support(ground_truth, predictions, average=None)
    accuracy_result = accuracy_score(ground_truth, predictions)
    f1_scores, precision, recall = results_actions[2], results_actions[0], results_actions[1]
    
    # printing metrics.
    print('Validation Epoch: %d, F1-Score: %s' % (epoch, str(f1_scores)))
    print('Validation Epoch: %d, accuracy result: %s' % (epoch, np.mean(accuracy_result)))
    print('Validation Epoch: %d, Loss: %.4f' % (epoch, np.mean(losses)))
    print('Validation Epoch: %d, F1-Score: %s' % (epoch, np.mean(f1_scores)))
    print('Validation Epoch: %d, precision_value: %.4f' % (epoch, np.mean(precision)))
    print('Validation Epoch: %d, recall_value: %4f' % (epoch, np.mean(recall)))

    # setting up tensorboard.
    writer.add_scalar('Validation Loss', np.mean(losses), epoch)
    writer.add_scalar('Validation Accuracy', np.mean(accuracy_result), epoch)
    writer.add_scalar('Validation F1-Score', np.mean(f1_scores), epoch)
    writer.add_scalar('Validation Precision', np.mean(precision), epoch)
    writer.add_scalar('Validation Recall', np.mean(recall), epoch)

