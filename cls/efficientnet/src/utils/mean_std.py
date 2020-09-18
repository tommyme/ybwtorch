# -*- coding: utf-8 -*-
import numpy as np
import cv2
import random
import os
from utils.data_pps import get_lists
import utils.config as config
from tqdm import tqdm

def calc_mean_std():
    path, _, _ = get_lists(config.root)
    path = path['train']+path['val']
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    print('default mean & std: \n', mean, std)
    mean = [0, 0, 0]
    std = [0, 0, 0]

    for each in tqdm(path):
        img = cv2.imread(each).astype(np.float32) / 255.
        for i in range(3):
            mean[i] += img[:, :, i].mean()
            std[i] += img[:, :, i].std()

    mean.reverse()
    std.reverse()

    mean = np.asarray(mean) / len(path)
    std = np.asarray(std) / len(path)
    print('our dataset: \n', mean, std)
    return mean,std



    

