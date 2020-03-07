import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import sys
sys.path.append('../src/')

from model_light import UNet
from SSNet import SSNetFinal
from dataset import SSSDataset
from loss import DiscriminativeLoss
from PIL import Image
import os

n_sticks = 8
max_n_sticks = 100
n_clusters = 8
random_n = True

# Model
model = UNet().cuda()
# model = SSNetFinal().cuda()

# Dataset for train
train_dataset = SSSDataset(train=True, n_sticks=n_sticks, max_n_sticks = max_n_sticks, random_n = random_n)
train_dataloader = DataLoader(train_dataset, batch_size=4,
                              shuffle=False, num_workers=6, pin_memory=True)

# Loss Function
criterion_disc = DiscriminativeLoss(delta_var=0.5,
                                    delta_dist=1.5,
                                    norm=2,
                                    usegpu=True).cuda()
criterion_ce = nn.CrossEntropyLoss().cuda()

# Optimizer
parameters = model.parameters()
optimizer = optim.SGD(parameters, lr=0.01, momentum=0.9, weight_decay=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 mode='min',
                                                 factor=0.1,
                                                 patience=10,
                                                 verbose=True)

# Train
model_dir = Path('../model')
model_path = model_dir.joinpath('model.pth')
if os.path.exists(model_path):
    param = torch.load(model_path)
    model.load_state_dict(param)

best_loss = np.inf
for epoch in range(300):
    print(f'epoch : {epoch}')
    disc_losses = []
    ce_losses = []
    for batched in train_dataloader:
        images, sem_labels, ins_labels, n_sticks_batch = batched
        images = Variable(images).cuda()
        sem_labels = Variable(sem_labels).cuda()
        ins_labels = Variable(ins_labels).cuda()
        model.zero_grad()

        sem_predict, ins_predict = model(images)
        loss = 0



        # Discriminative Loss
        if random_n: #unfix the number
            clusters = n_sticks_batch.data.data.cpu().numpy().astype(np.uint8)
            clusters = torch.Tensor(clusters)
            clusters = clusters.type(torch.cuda.FloatTensor)
            # clusters = [15, 28, 58, 68]
            disc_loss = criterion_disc(ins_predict,
                                   ins_labels,
                                   clusters)
                                   # n_sticks.type(torch.FloatTensor))
        else:
            disc_loss = criterion_disc(ins_predict,
                                   ins_labels,
                                   [n_clusters] * len(images))
        loss += disc_loss
        # disc_losses.append(disc_loss.cpu().data.numpy()[0])
        disc_losses.append(disc_loss.cpu().data.numpy())


        # Cross Entropy Loss
        _, sem_labels_ce = sem_labels.max(1)
        ce_loss = criterion_ce(sem_predict.permute(0, 2, 3, 1)\
                                   .contiguous().view(-1, 2),
                               sem_labels_ce.view(-1))
        loss += ce_loss
        # ce_losses.append(ce_loss.cpu().data.numpy()[0])
        ce_losses.append(ce_loss.cpu().data.numpy())

        loss.backward()
        optimizer.step()
    disc_loss = np.mean(disc_losses)
    ce_loss = np.mean(ce_losses)
    print(f'DiscriminativeLoss: {disc_loss:.4f}')
    print(f'CrossEntropyLoss: {ce_loss:.4f}')
    scheduler.step(disc_loss)
    if disc_loss < best_loss:
        best_loss = disc_loss
        print('Best Model!')
        modelname = 'model.pth'
        torch.save(model.state_dict(), model_dir.joinpath(modelname))                                                 