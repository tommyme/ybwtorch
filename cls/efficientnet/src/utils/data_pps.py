import os
import numpy as np
from shutil import copyfile
from random import shuffle
from glob import glob
import pandas as pd
import json
'''
file 文件名
paths 完整路径
label 数字
cls 单词
id 数字
'''


def get_lists(**kwargs):  # mydict指定***.txt的地址, idx < 5
    '''
    rags:
    1. idx用于bagging
    2. opt用于训练辅助分类器（本次有两个类别分不清）

    returns
    1. path 一个字典，分为'train'、'val'，他们分别包含着由绝对路径组成的列表
    2. label ... 由数字标签组成
    3. cls2id {cls_1:0, cls_2:1}
    '''
    paths = []
    path = {}
    label = {}
    for id in os.listdir('/content'):
        if id in ['1','2','3','4','5']:
            for file in os.listdir(os.path.join('/content',id)):
                paths.append(os.path.join('/content',id,file))
                # labels.append(str(int(id)-1))
    shuffle(paths)
    labels = [int(i.split('/')[-2])-1 for i in paths]
    split = int(0.8*len(paths))

    # if idx == -1:
    path['train'] = paths[:split]
    path['val'] = paths[split:]
    path['test'] = paths

    label['train'] = labels[:split]
    label['val'] = labels[split:]
    label['test'] = labels
    cls2id = ''

    return path, label, cls2id
