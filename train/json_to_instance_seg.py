import json
import math
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

# np.random.seed(0)
# matplotlib inline

import sys
sys.path.append('../src/')

from model_light import UNet
from dataset_line import SSSDataset
from utils import gen_color_img, gen_instance_mask,coloring,coloring_debug



def cal_distance(x1, y1, x2, y2):
	dist = math.sqrt((float(x2) - float(x1))**2 + (float(y2) - float(y1))**2)
	return dist

def merge_big_and_patch(img_big, instance_mask):
    unique_labels_big = np.unique(img_big)
    unique_labels_mask = np.unique(instance_mask)

    new_label = img_big.max()+1

    for mi in range(1,len(unique_labels_mask)):
        label_mask = unique_labels_mask[mi]
        pixels_mask = instance_mask[instance_mask == label_mask]
        len_max_freq_pixels = len(pixels_mask)
        pixels_big = img_big[instance_mask == label_mask]
        pixels_big = pixels_big[pixels_big!=0]
        if len(pixels_big) == 0:  #old big map is all zero
            img_big[instance_mask == label_mask] = new_label
            new_label = new_label+1
        else:
            pixels_big_unique, pixels_big_count = np.unique(pixels_big, return_counts = True)
            for pi in range(len(pixels_big_unique)):
                new_pix_len = len(img_big[img_big == pixels_big_unique[pi]])
                pixels_big_count[pi] = new_pix_len

            big_max_freq_pixels = pixels_big_count.max()
            if len_max_freq_pixels <= big_max_freq_pixels:
                label = pixels_big_unique[pixels_big_count == big_max_freq_pixels]
                label = label[0]
                img_big[instance_mask == label_mask] = label
            else:
                img_big[instance_mask == label_mask] = new_label
                new_label = new_label + 1


    return img_big

def reassign_big(img):
    unique_labels = np.unique(img)
    img_big = np.zeros_like(img)
    random_labels = [0] + list(np.random.permutation(len(unique_labels)-1)+1)

    for ui in range(1,len(unique_labels)):
        img_big[img == unique_labels[ui]] = random_labels[ui]
    return img_big

def filtering(img_big, threshold = 2):
    unique_labels, count = np.unique(img_big, return_counts=True)
    delete_labels = unique_labels[count <= threshold]
    for di in range(len(delete_labels)):
        img_big[img_big==delete_labels[di]] = 0
    return img_big

def main():
    json_file_path = './'

    # img_name = '20190403_CL4_EGFP-EPS8_mCh-Espin_005_Decon_MaxIP_crop1_256x256_frame1'
    # img_name = '20190403_CL4_EGFP-EPS8_mCh-Espin_005_Decon_MaxIP_crop2_256x256_frame9'
    img_name = 'simulation'

    patch_size = 128


    # Model
    model = UNet().cuda()
    model.eval()

    model_dir = Path('../model')
    model_path = model_dir.joinpath('model.pth')

    param = torch.load(model_path)
    model.load_state_dict(param)

    if img_name == 'simulation':
        n_sticks = 200
        max_n_sticks = 300
        # n_clusters = 20
        random_n = False
        test_dataset = SSSDataset(train=False, n_sticks=n_sticks, data_size=1, max_n_sticks=max_n_sticks,
                                  random_n=random_n)
        test_dataloader = DataLoader(test_dataset, batch_size=1,
                                     shuffle=False, num_workers=0,
                                     pin_memory=True)
        images = []
        sem_pred = []
        ins_pred = []

        with torch.no_grad():
            for images_, n_sticks_batch in test_dataloader:
                images_ = images_

        img = images_[0,0].data.cpu().numpy()
        count = n_sticks
    else:
        data = json.load(open(json_file_path + img_name+'.json'))
        img_rgb = cv2.imread(json_file_path + img_name+'.png')


        count = 0

        max_n_sticks = 200

        img = img_rgb[:, :, 0]
        img = np.zeros_like(img)

        gt = np.zeros_like(img)
        # gt = cv2.fillPoly(gt, [box], 1)

        for shape in data['shapes']:
            cor_set = shape['points']
            cor_set = np.array(cor_set, dtype=np.uint8)
            x1 = cor_set[0, 0]
            y1 = cor_set[0, 1]
            x2 = cor_set[1, 0]
            y2 = cor_set[1, 1]

            start_point = (x1, y1)
            end_point = (x2, y2)
            color = (255)
            thickness = 1
            gt = cv2.line(gt, start_point, end_point, color, thickness)
            img = cv2.line(img, start_point, end_point, color, thickness)
            count = count + 1

    # dist_set = []
    # distribution = np.zeros(31)


    patch_len_x = int(img.shape[0] / patch_size) * 2 - 1
    patch_len_y = int(img.shape[1] / patch_size) * 2 - 1
    img_big =  np.zeros_like(img)

    for pi in range(0,patch_len_x):
        for pj in range(0,patch_len_y):
            start_px = int(pi*(patch_size/2))
            end_px = start_px + patch_size
            start_py = int(pj*(patch_size/2))
            end_py = start_py + patch_size

            img_patch = img[start_px:end_px, start_py:end_py]

            images_ = np.zeros((1, 1, img_patch.shape[0], img_patch.shape[1]), dtype=np.float)
            images_[0, 0, :, :] = img_patch
            images_ = torch.Tensor(images_)

            # Inference
            images = []
            sem_pred = []
            ins_pred = []

            with torch.no_grad():
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

            instance_mask = gen_instance_mask(p_sem_pred[0], ins_pred[0], max_n_sticks)
            patch_big = np.zeros_like(img)
            patch_big[start_px:end_px, start_py:end_py] = instance_mask

            img_big = merge_big_and_patch(img_big, patch_big)
            img_big = reassign_big(img_big)
            img_big_rgb = coloring(img_big)
            cv2.imwrite(json_file_path + 'color_%d_%d.png'%(pi,pj),img_big_rgb)

    # img = cv2.resize(img, (128,128))
    img_big = filtering(img_big, threshold = 3)
    img_big = reassign_big(img_big)
    img_big_rgb = coloring(img_big)
    # img_big_rgb = coloring_debug(img_big)
    print('count true = %d, pred = %d' % (count,len(np.unique(img_big))))
    cv2.imwrite(json_file_path + img_name+'_binary.png', img)
    cv2.imwrite(json_file_path + img_name+'_rgb.png', img_big_rgb)

if __name__ == "__main__":
	main()

