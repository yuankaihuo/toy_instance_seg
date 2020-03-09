import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from PIL import Image
import time
import psutil


def gen_mask(ins_img):
    mask = []
    for i, mask_i in enumerate(ins_img):
        binarized = mask_i * (i + 1)
        mask.append(binarized)
    mask = np.sum(np.stack(mask, axis=0), axis=0).astype(np.uint8)
    return mask


def coloring(mask):
    ins_color_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    ins_black_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    n_ins = len(np.unique(mask)) - 1
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, n_ins)]
    for i in range(n_ins):
        ins_color_img[mask == i + 1] =\
            (np.array(colors[i][:3]) * 255).astype(np.uint8)
        ins_black_img[mask == i + 1] = \
            (np.array((0.5,0.5,0.5)) * 255).astype(np.uint8)
        ins_color_img_show = ins_black_img.copy()
        ins_color_img_show[mask == i + 1] =\
            (np.array(colors[i][:3]) * 255).astype(np.uint8)

        # test_image = Image.fromarray(ins_color_img_show)
        # test_image.show()
        # time.sleep(1)
        # # hide image
        # for proc in psutil.process_iter():
        #     if proc.name() == "display":
        #         proc.kill()

    return ins_color_img

def coloring_debug(mask):
    ins_color_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    ins_black_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    n_ins = len(np.unique(mask)) - 1
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, n_ins)]
    for i in range(n_ins):
        ins_color_img[mask == i + 1] =\
            (np.array(colors[i][:3]) * 255).astype(np.uint8)
        ins_black_img[mask == i + 1] = \
            (np.array((0.5,0.5,0.5)) * 255).astype(np.uint8)
        ins_color_img_show = ins_black_img.copy()
        ins_color_img_show[mask == i + 1] =\
            (np.array(colors[i][:3]) * 255).astype(np.uint8)

        test_image = Image.fromarray(ins_color_img_show)
        test_image.show()
        time.sleep(1)
        # hide image
        for proc in psutil.process_iter():
            if proc.name() == "display":
                proc.kill()

    return ins_color_img


def gen_instance_mask(sem_pred, ins_pred, n_obj):
    embeddings = ins_pred[:, sem_pred].transpose(1, 0)
    # clustering = KMeans(n_obj).fit(embeddings)
    clustering = MeanShift(bandwidth=0.5).fit(embeddings)
    print('unique sticks = %d' % len(np.unique(clustering.labels_)))
    labels = clustering.labels_

    instance_mask = np.zeros_like(sem_pred, dtype=np.uint8)
    for i in range(n_obj):
        lbl = np.zeros_like(labels, dtype=np.uint8)
        lbl[labels == i] = i + 1
        instance_mask[sem_pred] += lbl

    return instance_mask


def gen_color_img(sem_pred, ins_pred, n_obj):
    return coloring(gen_instance_mask(sem_pred, ins_pred, n_obj))
