
from __future__ import print_function, division

from functions import *
from train import *
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


plt.ion()   # interactive mode


# DATA PREP
dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = dir_path + '/vinyl-dataset'
print(data_dir)
data_transforms = define_data_transforms()
(image_datasets, dataloaders, dataset_sizes, class_names) = prepare_data(data_transforms, data_dir)


# USE GPU IF AVAILABLE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# SHOW SOME IMGS
#inputs, classes = next(iter(dataloaders['train'])) # Get a batch of training data
#out = torchvision.utils.make_grid(inputs, nrow=4) # Make a grid from batch
#imshow(out, title=[class_names[x] for x in classes])


# LOAD PRETRAINED MODEL
resnet18 = models.resnet18(pretrained=True) # Load pretrained resnet18 model
#for param in resnet18.parameters(): # freeze all weights, except for last fc
#    param.requires_grad = False
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 3)
#resnet18.fc = nn.Sequential(nn.Linear(num_ftrs, 3)) # Add dropout to final fc layer
resnet18 = resnet18.to(device) # Move model to GPU (or CPU, if you're GPU i whack!)
	

# SET HYPERPARAMETERS
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# TRAINE THE MODEL
resnet18 = train_model(resnet18, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, device, dataset_sizes,
                       num_epochs=25)

# SAVE MODEL
print("\nSaving model..")
torch.save(resnet18.state_dict(), "/home/kasper/code/vinyl/LOGDATA_saved_model.pt")


#PLOT ALL PLOTS
print("Plotting..")
plt.show(block=True)

