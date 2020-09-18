#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import os
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.dataloader import efficientdet_dataset_collate, EfficientdetDataset
from nets.efficientdet import EfficientDetBackbone
from nets.efficientdet_training import Generator, FocalLoss
from tqdm import tqdm
from config import classes
import argparse
from torch.utils.model_zoo import load_url

model_urls = [
            'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d0.pth',
            'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d1.pth',
            'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d2.pth',
            'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d3.pth',
            'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d4.pth',
            'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d5.pth',
            'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d6.pth',
            'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.2/efficientdet-d7.pth',
            'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.2/efficientdet-d8.pth'
            ]

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   获得类和先验框
#---------------------------------------------------#

def fit_one_epoch(net,focal_loss,epoch,epoch_size,epoch_size_val,gen,genval,Epoch):
    total_r_loss = 0
    total_c_loss = 0
    total_loss = 0
    val_loss = 0
    start_time = time.time()
    with tqdm(total=epoch_size,desc=f'Train Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets]

            optimizer.zero_grad()
            _, regression, classification, anchors = net(images)
            loss, c_loss, r_loss = focal_loss(classification, regression, anchors, targets, cuda=True)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_r_loss += r_loss.item()
            total_c_loss += c_loss.item()
            waste_time = time.time() - start_time
            
            pbar.set_postfix(**{'Conf Loss'         : total_c_loss / (iteration+1), 
                                'Regression Loss'   : total_r_loss / (iteration+1), 
                                'lr'                : get_lr(optimizer),
                                'step/s'            : waste_time})
            pbar.update(1)

            start_time = time.time()


    with tqdm(total=epoch_size_val, desc=f'Val Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets_val]
                optimizer.zero_grad()
                _, regression, classification, anchors = net(images_val)
                loss, c_loss, r_loss = focal_loss(classification, regression, anchors, targets_val, cuda=True)
                val_loss += loss.item()

            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)
            
    print('\nEpoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    
    return total_loss/(epoch_size+1), val_loss/(epoch_size_val+1), model.state_dict()
#----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--lr", type=float, help="learning_rate", default=1e-3)
    parser.add_argument("-f", "--freeze", type=int, help="frozen_epoches", default=5)
    parser.add_argument("-u", "--unfreeze", type=int, help="free_epoches", default=5)
    parser.add_argument('-v', "--version", type=int, help='you want to use efficientdet-dX?', default=0)
    parser.add_argument('-b', "--batch_size", type=int, help='just batch_size', default=64)
    parser.add_argument('-p', "--pre_model", type=str, help='just pretrained_model_path default = \'\'', default='')
    parser.add_argument('-s', '--val_split', type=float, help='验证集的比例', default=0.1)
    
    args = parser.parse_args()
    
    phi = args.version
    annotation_path = '2007_train.txt' 
    num_classes = len(classes)

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1408, 1536]
    input_shape = (input_sizes[phi], input_sizes[phi])

    # 创建模型
    model = EfficientDetBackbone(num_classes,phi)
    # 加快模型训练的效率
    print('Loading weights into state dict...')
    model_dict = model.state_dict() 
    pretrained_dict = torch.load(args.pre_model) if args.pre_model else load_url(model_urls[phi]) 
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    net = model.train()
    net = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    net = net.cuda()

    efficient_loss = FocalLoss()

    # 0.1用于验证，0.9用于训练
    val_split = args.val_split
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val


    # 初始化参数
    lr = args.lr
    Batch_size = args.batch_size
    Init_Epoch = int(args.pre_model.split('/')[-1].split('-')[0][5:]) if args.pre_model else 0
    Freeze_Epoch = args.freeze
    Unfreeze_Epoch = args.unfreeze
    
    # 优化器
    optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    # generator
    train_dataset = EfficientdetDataset(lines[:num_train], (input_shape[0], input_shape[1]))
    val_dataset = EfficientdetDataset(lines[num_train:], (input_shape[0], input_shape[1]))
    gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                            drop_last=True, collate_fn=efficientdet_dataset_collate)
    gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                            drop_last=True, collate_fn=efficientdet_dataset_collate)


    epoch_size = num_train//Batch_size
    epoch_size_val = num_val//Batch_size
    
    
    val_losses = []
    train_losses = []
    #解冻一部分
    for param in model.backbone_net.parameters():
        param.requires_grad = False
    print('freezed !')
    for epoch in range(Init_Epoch,Freeze_Epoch+Unfreeze_Epoch):
        # 全部解冻
        if epoch == Freeze_Epoch:
            print('unfreezed !')
            lr = lr / 10
            Batch_size = int(Batch_size / 4)
            for param in model.backbone_net.parameters():
                param.requires_grad = True
            
            gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=efficientdet_dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                    drop_last=True, collate_fn=efficientdet_dataset_collate)
        # 保存最优权重的逻辑（内存不够，所以保存最优）
        total_loss, val_loss, weight = fit_one_epoch(net,efficient_loss,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch+Unfreeze_Epoch)
        lr_scheduler.step(val_loss)
        val_losses.append(val_loss)
        train_losses.append(total_loss)
        if max(val_losses) == val_loss:
            print('\n*************************** best val_loss, saving... ***************************\n')
            torch.save(weight, './Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))