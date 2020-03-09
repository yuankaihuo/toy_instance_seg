import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from PIL import Image
import random


class SSSDataset(Dataset):
    def __init__(self, train, n_sticks=8, data_size=512, max_n_sticks = 8, random_n=False, img_size=256):
        super().__init__()
        self.train = train
        self.n_sticks = n_sticks
        self.max_n_sticks = max_n_sticks
        self.data_size = data_size
        self.height = img_size
        self.width = img_size
        self.random_n = random_n

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        while True:
            img = np.ones((self.height, self.width), dtype=np.uint8) * 255
            ins = np.zeros((self.max_n_sticks, self.height, self.width), dtype=np.uint8)
            stick_num = 0
            if self.random_n:
                n_sticks = random.randint(2,self.max_n_sticks)
            else:
                n_sticks = self.n_sticks

            for _ in range(self.max_n_sticks):
                # print(stick_num)
                x = np.random.randint(15, self.height-15)
                y = np.random.randint(15, self.width-15)
                w = 1
                # h = np.random.randint(80, 100)
                h = np.random.randint(10, 30)
                theta = random.uniform(-90,90)
                rect = ([x, y], [w, h], theta)      # 中心(x,y), (宽,高), 旋转角度
                box = np.int0(cv2.boxPoints(rect))
                start_point = tuple(box[0])
                end_point = tuple(box[1])
                color = (0)
                thickness = 1

                if stick_num < n_sticks:
                    gt = np.zeros_like(img)
                    gt = cv2.line(gt, start_point, end_point, 1, thickness)

                    ins[:, gt != 0] = 0
                    ins[stick_num, gt>0] = 1
                    img = cv2.line(img, start_point, end_point, color, thickness)

                stick_num = stick_num + 1


            # minimum area of stick
            if np.sum(np.sum(ins, axis=(1, 2)) < 1) == (self.max_n_sticks - n_sticks):
                break

            # break

        if self.train:
            sem = np.zeros_like(img, dtype=bool)
            sem[np.sum(ins, axis=0) != 0] = True
            img_fill = torch.Tensor(sem[np.newaxis] * 255)
            sem = np.stack([~sem, sem]).astype(np.uint8)
            # Image.fromarray(sem[0].data.cpu().numpy() * 255).show()


            # 1 * height * width
            img = torch.Tensor(img[np.newaxis])
            # 2 * height * width
            sem = torch.Tensor(sem)
            # n_sticks * height * width
            ins = torch.Tensor(ins)
            # return img, sem, ins
            return img_fill, sem, ins, n_sticks
        else:
            # 1 * height * width
            sem = np.zeros_like(img, dtype=bool)
            # img = torch.Tensor(img[np.newaxis])
            sem[np.sum(ins, axis=0) != 0] = True
            img_fill = torch.Tensor(sem[np.newaxis] * 255)
            # sem = np.stack([~sem, sem]).astype(np.uint8)
            # return img
            return img_fill, n_sticks
