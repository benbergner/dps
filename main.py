#!/usr/bin/env python

import os
import numpy as np
import sklearn
import torch
from torch import nn

from utils import adjust_learning_rate, adjust_sigma
from traffic_dataset import TrafficSigns
from dps import DPS

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Settings

# Data
data_dir = 'traffic_dataset'
num_workers = 16
n_channel = 3
n_class = 4
patch_size = 100
high_size = (1200, 1600)
down_factor = 3
low_size = (high_size[0] // down_factor, high_size[1] // down_factor)

# General
n_epoch = 150
batch_size = 16

# Opt
n_epoch_warm_up = 10
lr = 1e-4
weight_decay = 1e-1
max_grad_norm = 0.1

# DPS specific
k = 2
num_samples = 500
sigma = 0.05
score_size = (24, 32)

# Transformer specific
n_layer = 1
n_token = [1]
d_model = 512
n_head = 8
d_k = 64
d_v = 64
d_inner = 512
attn_dropout = 0.
dropout = 0.1

###

# Define datasets and dataloaders
train_data = TrafficSigns(data_dir, high_size, low_size, train=True)
test_data = TrafficSigns(data_dir, high_size, low_size, train=False)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# Init the DPS model
net = DPS(n_class, n_channel, high_size, score_size, k, num_samples, sigma, patch_size,
    n_layer, n_token, n_head, d_k, d_v, d_model, d_inner, dropout, attn_dropout, device).to(device)

# Compute class weights as the dataset is imbalanced
all_train_labels = [instance[1] for instance in train_data._data]
class_weights = sklearn.utils.class_weight.compute_class_weight('balanced', classes=np.unique(all_train_labels), y=all_train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Define loss function and optimizer
loss_fn = nn.NLLLoss(weight=class_weights)
optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

for epoch in range(n_epoch):

    loss_ls = []
    correct, total = 0, 0
    
    # Training 
    net.train()

    for data_it, data in enumerate(train_loader, start=epoch * len(train_loader)):
        images_high, images_low, labels = data[0].to(device), data[1].to(device), data[2].to(device)

        adjust_learning_rate(n_epoch_warm_up, n_epoch, lr, optimizer, train_loader, data_it+1)
        adjust_sigma(0, n_epoch, 0.05, net, train_loader, data_it+1)
        optimizer.zero_grad()

        # get prediction
        pred = net(images_high, images_low)

        # compute loss and gradients
        loss = loss_fn(torch.log(pred + 1e-6), labels)
        loss.backward()

        # optimize
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
        optimizer.step()

        loss_ls.append(loss.item())
        pred_class = pred.max(dim = -1)[1]
        correct += (pred_class == labels).sum()
        total += pred_class.shape[0]

    mean_loss = np.mean(loss_ls)
    accuracy = correct / total
    current_lr = optimizer.param_groups[0]['lr']

    print("Train Epoch: {}, lr: {}, sigma: {}, loss: {}, accuracy: {}".format(epoch+1, current_lr, net.TOPK.sigma, mean_loss, accuracy))

    loss_ls = []
    correct, total = 0, 0

    # Evaluation
    net.eval()
    with torch.no_grad():
        for data in test_loader:
            images_high, images_low, labels = data[0].to(device), data[1].to(device), data[2].to(device)

            pred = net(images_high, images_low)

            loss = loss_fn(torch.log(pred + 1e-6), labels)

            loss_ls.append(loss.item())
            pred_class = pred.max(dim = -1)[1]

            correct += (pred_class == labels).sum()
            total += pred_class.shape[0]

    mean_loss = np.mean(loss_ls)
    accuracy = correct / total

    print("Test Epoch: {}, loss: {}, accuracy: {}".format(epoch + 1, mean_loss, accuracy))