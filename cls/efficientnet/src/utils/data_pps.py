import os
import numpy as np
from shutil import copyfile
from random import shuffle
from glob import glob
import pandas as pd
import json
import random

'''
file 文件名
paths 完整路径
label 数字
cls 单词
id 数字
'''
random.seed(123)


j = os.path.join
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
    root_test = 'test/test_set'
    pathtest = [j(root_test,i) for i in os.listdir(root_test)]
    paths = []
    path = {}
    label = {}
    cls2id = {'sunny':0,'cloudy':1,'others':2}
    root = '/content'
    for cls in ['sunny','cloudy','others']:
        diri = j(root,cls)
        for each in os.listdir(diri):
            paths.append(j(root,cls,each))
    shuffle(paths)
    labels = [cls2id[i.split('/')[-2]] for i in paths]
    split = int(0.8*len(paths))

    # if idx == -1:
    path['train'] = paths[:split]
    path['val'] = paths[split:]
    path['test'] = pathtest

    label['train'] = labels[:split]
    label['val'] = labels[split:]
    label['test'] = [0]*len(pathtest)


    return path, label, cls2id
