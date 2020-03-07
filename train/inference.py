import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage as ndi

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image

np.random.seed(0)
# matplotlib inline

import sys
sys.path.append('../src/')

from model_light import UNet
from dataset import SSSDataset
from utils import gen_color_img

n_sticks = 60
max_n_sticks = 100
# n_clusters = 20
random_n = False

# Model
model = UNet().cuda()
model.eval()

model_dir = Path('../model')
model_path = model_dir.joinpath('model.pth')

param = torch.load(model_path)
model.load_state_dict(param)

# Dataset for inference
test_dataset = SSSDataset(train=False, n_sticks=n_sticks, data_size=16, max_n_sticks = max_n_sticks, random_n = random_n)
test_dataloader = DataLoader(test_dataset, batch_size=16,
                             shuffle=False, num_workers=0,
                             pin_memory=True)

# Inference
images = []
sem_pred = []
ins_pred = []

with torch.no_grad():
    for images_, n_sticks_batch in test_dataloader:
        images.append(images_.numpy())
        images_ = Variable(images_).cuda()
        sem_pred_, ins_pred_ = model(images_)
        sem_pred.append(F.softmax(sem_pred_, dim=1).cpu().data.numpy())
        ins_pred.append(ins_pred_.cpu().data.numpy())

images = np.concatenate(images)[:, 0].astype(np.uint8)
sem_pred = np.concatenate(sem_pred)[:, 1, :, :]
ins_pred = np.concatenate(ins_pred) 

# Post Processing
p_sem_pred = []
for sp in sem_pred:
    # p_sem_pred.append(ndi.morphology.binary_fill_holes(sp > 0.5))
    p_sem_pred.append(sp > 0.5)

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
plt.gray()
plt.savefig('gray.png')

for i, ax_ in enumerate(axes):
    color_img = gen_color_img(p_sem_pred[i], ins_pred[i], n_sticks)
    ax_[0].imshow(images[i])
    plt.savefig('ax0'+str(i)+'.png')
    ax_[1].imshow(~p_sem_pred[i])
    plt.savefig('ax1'+str(i)+'.png')
    ax_[2].imshow(color_img)
    plt.savefig('ax2'+str(i)+'.png')                          