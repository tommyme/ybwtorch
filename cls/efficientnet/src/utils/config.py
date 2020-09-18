from torchvision import transforms
import numpy as np
import torch
from PIL import Image
import torchvision


from utils.auto_augment import AutoAugment
import pandas as pd


class Resize_propotion(object):
    """
    等比缩放，防止在缩放过程中导致物体长宽比例改变导致识别错误
    """
    def __init__(self,size,interpolation = Image.BILINEAR):
        self.size = (size,size)
        self.interpolation =interpolation




    def __call__(self,img):
        #padding
        ratio = self.size[0] / self.size[1]
        w,h = img.size
        if w/h < ratio:
            t = int(h * ratio)
            w_padding = (t-w)//2
            img = img.crop((-w_padding,0,w+w_padding,h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h + h_padding))

        img = img.resize(self.size, self.interpolation)
        return img

# [0.26958571,0.3887648,0.41509606] [0.16806719,0.18439094,0.18346951]
mean = np.array([0.26958571,0.3887648,0.41509606])
std = np.array([0.16806719,0.18439094,0.18346951])




batch_size = 20
num_workers = 4
label_smooth = True
lr = 1e-3
cbam = 0
version = 4
pre_model = None
mean_std = False
confusion_matrix=True
# 下面的参数一般要改
input_size = [224,240,260,300,380,456,528,600][version]
num_classes = 20
root = '/content/dataset' # 传给data_pps的参数
model_path = None # 预训练模型的位置 默认为None
out_dir = './' # 路径后面不能有斜杠   '%s/net_%03d_%.3f.pth' % (config.outdir, epoch + 1,acc))

use_sample_weight = True
train_df = pd.read_csv('/content/dataset/training.csv')
train_dist = train_df.groupby('SpeciesID').count()
avg_num = len(train_df) / 20
sample_weight = {i : avg_num/int(train_dist.loc[i,:]) for i in range(num_classes)}

test_transform = transforms.Compose([
    torchvision.transforms.Resize(input_size),
    torchvision.transforms.CenterCrop(input_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),
])

train_transform = transforms.Compose([
    torchvision.transforms.Resize(input_size),# 要不要加随即裁剪呢
    torchvision.transforms.CenterCrop(input_size),
    # Resize_propotion(input_size),
    torchvision.transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    torchvision.transforms.RandomHorizontalFlip(),
    # AutoAugment(dataset='IMAGENET'),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),
])

trans_tta_1 = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
trans_tta_2 = transforms.Compose([
    transforms.Resize(input_size),
    transforms.RandomRotation((-25, 25)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
trans_tta_3 = transforms.Compose([
    transforms.Resize(input_size),
    transforms.RandomHorizontalFlip(1),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
trans_tta_4 = transforms.Compose([
    transforms.Resize(input_size),
    transforms.RandomVerticalFlip(1),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
trans_tta_5 = transforms.Compose([
    Resize_propotion(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


tta_trans_list = [trans_tta_1,trans_tta_2,trans_tta_3,trans_tta_4,trans_tta_5]