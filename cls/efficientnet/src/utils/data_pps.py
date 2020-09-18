import os
import numpy as np
from shutil import copyfile
from random import shuffle
from glob import glob
import pandas as pd

'''
file 文件名
paths 完整路径
label 数字
cls 单词
id 数字
'''

def get_lists(root,idx=-1,opt=False):# mydict指定***.txt的地址, idx < 5
    
    '''
    rags:
    1. idx用于bagging
    2. opt用于训练辅助分类器（本次有两个类别分不清）
    
    returns
    1. path 一个字典，分为'train'、'val'，他们分别包含着由绝对路径组成的列表
    2. label ... 由数字标签组成
    3. cls2id {cls_1:0, cls_2:1}
    '''
    
    # 本次针对海洋生物数据集
    species = pd.read_csv(os.path.join(root,'species.csv'))
    cls2id = {list(species.ScientificName)[i]:list(species.ID)[i] for i in range(species.shape[0])}

    data = pd.read_csv(os.path.join(root,'training.csv'))
    if opt:
        data = data[data['SpeciesID'].isin([0,4])]
    data = data.sample(frac=1, random_state=123)
    paths2train = [root+'/data/'+i+'.jpg' for i in list(data['FileID'])]
    labels2train = list(data['SpeciesID'])

    anno = pd.read_csv(os.path.join(root,'annotation.csv')) # test.csv文件名和真实标签，用于验证
    if opt:
        anno = anno[anno['SpeciesID'].isin([0,4])]
    paths4test = [root+'/data/'+i+'.jpg' for i in list(anno['FileID'])]
    labels4test = list(anno['SpeciesID'])

    path = {}
    label = {}
    
    split = int(0.8*len(paths2train))
    split_n = [int(i*len(paths2train)) for i in [0,0.2,0.4,0.6,0.8,1]]
    
    if idx == -1:
        path['train'] = paths2train[:split]
        path['val'] = paths2train[split:]
        path['test'] = paths4test

        label['train'] = labels2train[:split]
        label['val'] = labels2train[split:]
        label['test'] = labels4test
    else :
        path['train'] = paths2train[:split_n[idx]] + paths2train[split_n[idx+1]:]
        path['val'] = paths2train[split_n[idx]:split_n[idx+1]]
        path['test'] = paths4test

        label['train'] = labels2train[:split_n[idx]] + labels2train[split_n[idx+1]:]
        label['val'] = labels2train[split_n[idx]:split_n[idx+1]]
        label['test'] = labels4test
        
    return path,label,cls2id






