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
# general training settings
n_epoch = 150
b = 16
num_workers = 16
warm_up_lr = 1e-4
n_epoch_warm_up = 10
lr = 1e-4
weight_decay = 1e-1
k = 10
n_class = 4
n_channel = 3
patch_size = 100
img_size = (1200, 1600)
down_factor = 3
low_size = (img_size[0] // down_factor, img_size[1] // down_factor)

# cross-attention specific
n_layer = 1
n_token = [1]
d_model = 512
n_head = 8
d_k = 64
d_v = 64
d_inner = 2048
attn_dropout = 0.1
dropout = 0.1

data_dir = '/dhc/home/benjamin.bergner/netstore-old/data/speedlimits'
train_data = TrafficSigns(data_dir, low_size, train=True)
test_data = TrafficSigns(data_dir, low_size, train=False)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=b, shuffle=True, num_workers=num_workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=b, shuffle=False, num_workers=num_workers, pin_memory=True)

#n_class, n_channel, k, patch_size, n_layer, n_token,
#n_head, d_k, d_v, d_model, d_inner, dropout, attn_dropout, device

net = DPS(n_class, n_channel, k, patch_size, n_layer, n_token, n_head, 
    d_k, d_v, d_model, d_inner, dropout, attn_dropout, device).to(device)

all_train_labels = [instance[1] for instance in train_data._data]
class_weights = sklearn.utils.class_weight.compute_class_weight('balanced', classes=np.unique(all_train_labels), y=all_train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

loss_fn = nn.NLLLoss(weight=class_weights)
optimizer = torch.optim.AdamW(net.parameters(), lr=warm_up_lr, weight_decay=weight_decay)

for epoch in range(n_epoch):

    loss_ls = []
    loss_entr_ls = []
    correct, total = 0, 0

    net.train()

    for data_it, data in enumerate(train_loader, start=epoch * len(train_loader)):
        images_high, images_low, labels = data[0].to(device), data[1].to(device), data[2].to(device)
        #print("images_high: ", images_high.shape)
        #print("images_low: ", images_low.shape)

        adjust_learning_rate(n_epoch_warm_up, n_epoch, lr, optimizer, train_loader, data_it+1)
        adjust_sigma(0, n_epoch, 0.05, net, train_loader, data_it+1)
        optimizer.zero_grad()

        pred, entr = net(images_high, images_low)

        loss = loss_fn(torch.log(pred + 1e-6), labels)
        loss_entr = 0.01 * entr
        (loss).backward()#loss_entr

        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
        optimizer.step()

        loss_ls.append(loss.item())
        loss_entr_ls.append(loss_entr.item())
        pred_class = pred.max(dim = -1)[1]
        correct += (pred_class == labels).sum()
        total += pred_class.shape[0]

    mean_loss = np.mean(loss_ls)
    mean_loss_entr = np.mean(loss_entr_ls)
    accuracy = correct / total
    print("Train Epoch: {}, lr: {}, loss: {}, loss entr: {}, accuracy: {}".format(epoch+1, optimizer.param_groups[0]['lr'], mean_loss, mean_loss_entr, accuracy))
    print("sigma: ", net.TOPK.sigma)

    #eval
    loss_ls = []
    loss_entr_ls = []
    correct, total = 0, 0

    net.eval()
    with torch.no_grad():
        for data in test_loader:
            images_high, images_low, labels = data[0].to(device), data[1].to(device), data[2].to(device)

            pred, entr = net(images_high, images_low)

            loss = loss_fn(torch.log(pred + 1e-6), labels)
            loss_entr = 0.01 * entr

            loss_ls.append(loss.item())
            loss_entr_ls.append(loss_entr.item())
            pred_class = pred.max(dim = -1)[1]

            correct += (pred_class == labels).sum()
            total += pred_class.shape[0]

    mean_loss = np.mean(loss_ls)
    mean_loss_entr = np.mean(loss_entr_ls)
    accuracy = correct / total

    print("Test Epoch: {}, loss: {}, loss_entr: {}, accuracy: {}".format(epoch + 1, mean_loss, mean_loss_entr, accuracy))