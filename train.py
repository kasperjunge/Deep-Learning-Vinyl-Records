import torch
import torch.nn as nn
from torch.nn import Dropout
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
import inspect
from pandas_ml import ConfusionMatrix
import pickle


def train_model(model, criterion, optimizer, scheduler, dataloaders, device, dataset_sizes, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # FOR PLOTTING LEARNING CURVE
    epoch_axis = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    log_train_loss = []
    log_train_acc = []
    log_val_loss = []
    log_val_acc = []

    #FOR PLOTTING CONF. MATRIX
    preds_temp = []
    labels_temp = []
    preds_best = []
    labels_best = []


    for epoch in range(num_epochs):
        print('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        count = 0

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0


            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()


                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    #loss = criterion(preds, labels) # For MSELoss()

                    #print(outputs)
                    #print(preds)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                #print(preds.cpu().numpy()[0])
                #print(preds.cpu().numpy()[1])
                # PREPARE CONF. MATRIX
                for x in range(len(labels)):
                    labels_temp.append(labels.cpu().numpy()[x])

                for x in range(len(preds)):
                    preds_temp.append(preds.cpu().numpy()[x])


                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)



            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train': # Saving training loss and acc for later plotting
                log_train_loss.append(epoch_loss)
                log_train_acc.append(epoch_acc)
                #print(log_train_loss)
                #print(log_train_acc)
            
            if phase == 'val': # Saving validation loss and acc for later plotting
                log_val_loss.append(epoch_loss)
                log_val_acc.append(epoch_acc)
                #print(log_val_loss)
                #print(log_val_acc)

            if phase == 'val' and epoch_acc > best_acc: # deep copy the model + saving best preds and labels
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                preds_best = preds_temp
                labels_best = labels_temp
                

        
    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))



    # PREPARE TRAINING CURVE PLOT
    # Define data to be plotted
    print("\nPlotting training curve..")
    df = pd.DataFrame({ 
    'epoch_axis': epoch_axis,
    'val_acc': log_val_acc,
    'val_loss': log_val_loss,
    'train_acc': log_train_acc,
    'train_loss': log_train_loss
    })
    
    # multiple line plot
    plt.plot( 'epoch_axis', 'val_acc', data=df, marker='' , color='blue', linewidth=1)
    plt.plot( 'epoch_axis', 'val_loss', data=df, marker='', color='red', linewidth=1)
    plt.plot( 'epoch_axis', 'train_acc', data=df, marker='' , color='blue', linewidth=1, linestyle='dashed')
    plt.plot( 'epoch_axis', 'train_loss', data=df, marker='', color='red', linewidth=1, linestyle='dashed')
    plt.legend(('Val. Acc.', 'Val. Loss', 'Train Acc.', 'Train Loss'))


    # PREPARE CONFUSION MATRIX PLOT 
    print("\nPlotting confusion matrix.")
    #confusion_matrix = ConfusionMatrix(labels_best, preds_best)
    confusion_matrix = ConfusionMatrix(labels_best, preds_best, labels=['60s', '70s', '80s'])
    confusion_matrix.plot(normalized=True)
    print("Confusion matrix:\n%s" % confusion_matrix)


    # SAVE PREDS+LABELS BEST VECTORS
    with open('LOGDATA_preds_best.pkl', 'wb') as pickle_file:
        pickle.dump(preds_best, pickle_file)

    with open('LOGDATA_labels_best.pkl', 'wb') as pickle_file:
        pickle.dump(labels_best, pickle_file)


    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
