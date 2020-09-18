from torchvision import datasets, transforms
import torch
import torchvision
from efficientnet_pytorch import EfficientNet
import os
from utils.config import test_transform, num_classes, root, model_path
from PIL import Image
import numpy as np
import time
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed",type=int,default=123,help='path = os.listdir(root)[seed]')
args = parser.parse_args()

net = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
net.load_state_dict(torch.load(model_path,torch.device('cpu')))
net.eval()




test_data = pd.read_csv('/content/dataset/annotation.csv')

start = time.time()
filename = test_data.iloc[args.seed,0] + '.jpg'
gt_label = test_data.iloc[args.seed,1]
image = Image.open(os.path.join(root,filename)).convert('RGB')
print('filename',filename)
input = test_transform(image).unsqueeze(0)
output = net(input)

print('predicted label:',int(output.argmax()))
print('gt label:',gt_label)
print('用时',1000*(time.time()-start),'ms')