import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.models as models
import numpy as np 
from tqdm import tqdm
from torchvision import models
from torch.autograd import Variable
from PIL import Image
from torch import nn
from nets.pspnet import PSPNet
from nets.pspnet_training import CE_Loss,Dice_loss
from utils.metrics import f_score
from torch.utils.data import DataLoader
from utils.dataloader import pspnet_dataset_collate, PSPnetDataset


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(net,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda,aux_branch):
    net = net.train()
    total_loss = 0
    total_f_score = 0

    val_toal_loss = 0
    val_total_f_score = 0
    start_time = time.time()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size: 
                break
            imgs, pngs, labels = batch

            with torch.no_grad():
                imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
                pngs = Variable(torch.from_numpy(pngs).type(torch.FloatTensor)).long()
                labels = Variable(torch.from_numpy(labels).type(torch.FloatTensor))
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()

            #-------------------------------#
            #   判断是否使用辅助分支并回传
            #-------------------------------#
            optimizer.zero_grad()
            if aux_branch:
                aux_outputs, outputs = net(imgs)
                aux_loss  = CE_Loss(aux_outputs, pngs, num_classes = NUM_CLASSES)
                main_loss = CE_Loss(outputs, pngs, num_classes = NUM_CLASSES)
                loss      = aux_loss + main_loss
                if dice_loss:
                    aux_dice  = Dice_loss(aux_outputs, labels)
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + aux_dice + main_dice

            else:
                outputs = net(imgs)
                loss    = CE_Loss(outputs, pngs, num_classes = NUM_CLASSES)
                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + main_dice

            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_f_score += _f_score.item()
            
            waste_time = time.time() - start_time
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                's/step'    : waste_time,
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

            start_time = time.time()

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            imgs, pngs, labels = batch
            with torch.no_grad():
                imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
                pngs = Variable(torch.from_numpy(pngs).type(torch.FloatTensor)).long()
                labels = Variable(torch.from_numpy(labels).type(torch.FloatTensor))
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()
                #-------------------------------#
                #   判断是否使用辅助分支
                #-------------------------------#
                if aux_branch:
                    aux_outputs, outputs = net(imgs)
                    aux_loss  = CE_Loss(aux_outputs, pngs, num_classes = NUM_CLASSES)
                    main_loss = CE_Loss(outputs, pngs, num_classes = NUM_CLASSES)
                    val_loss  = aux_loss + main_loss
                    if dice_loss:
                        aux_dice  = Dice_loss(aux_outputs, labels)
                        main_dice = Dice_loss(outputs, labels)
                        val_loss  = val_loss + aux_dice + main_dice

                else:
                    outputs  = net(imgs)
                    val_loss = CE_Loss(outputs, pngs, num_classes = NUM_CLASSES)
                    if dice_loss:
                        main_dice = Dice_loss(outputs, labels)
                        val_loss  = val_loss + main_dice

                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)

                val_toal_loss += val_loss.item()
                val_total_f_score += _f_score.item()
                
            
            pbar.set_postfix(**{'total_loss': val_toal_loss / (iteration + 1),
                                'f_score'   : val_total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
            
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_toal_loss/(epoch_size_val+1)))

    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_toal_loss/(epoch_size_val+1)))



if __name__ == "__main__":
    inputs_size = [473,473,3]
    log_dir = "logs/"   
    #---------------------#
    #   分类个数+1
    #   2+1
    #---------------------#
    NUM_CLASSES = 2
    #--------------------------------------------------------------------#
    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    #---------------------------------------------------------------------# 
    dice_loss = True
    #-------------------------------#
    #   主干网络预训练权重的使用
    #   mobilenet和resnet50
    #-------------------------------#
    pretrained = False
    backbone = "mobilenet"
    #---------------------#
    #   是否使用辅助分支
    #   会占用大量显存
    #---------------------#
    aux_branch = False
    #---------------------#
    #   下采样的倍数
    #   8和16
    #---------------------#
    downsample_factor = 16
    #-------------------------------#
    #   Cuda的使用
    #-------------------------------#
    Cuda = True

    model = PSPNet(num_classes=NUM_CLASSES, backbone=backbone, downsample_factor=downsample_factor, pretrained=pretrained, aux_branch=aux_branch).train()
    
    # voc数据集下进行训练的
    model_path = r"model_data/pspnet_mobilenetv2.pth"
    # 加快模型训练的效率
    print('Loading weights into state dict...')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # 打开数据集的txt
    with open(r"VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt","r") as f:
        train_lines = f.readlines()

    # 打开数据集的txt
    with open(r"VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt","r") as f:
        val_lines = f.readlines()
        
    if True:
        lr = 1e-4
        Init_Epoch = 0
        Interval_Epoch = 50
        Batch_size = 8
        optimizer = optim.Adam(model.parameters(),lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)

        train_dataset = PSPnetDataset(train_lines, inputs_size, NUM_CLASSES, True)
        val_dataset = PSPnetDataset(val_lines, inputs_size, NUM_CLASSES, False)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=1, pin_memory=True,
                                drop_last=True, collate_fn=pspnet_dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                drop_last=True, collate_fn=pspnet_dataset_collate)

        epoch_size      = max(1, len(train_lines)//Batch_size)
        epoch_size_val  = max(1, len(val_lines)//Batch_size)

        for param in model.backbone.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch,Interval_Epoch):
            fit_one_epoch(model,epoch,epoch_size,epoch_size_val,gen,gen_val,Interval_Epoch,Cuda,aux_branch)
            lr_scheduler.step()
    
    if True:
        lr = 1e-5
        Interval_Epoch = 50
        Epoch = 100
        Batch_size = 4
        optimizer = optim.Adam(model.parameters(),lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)

        train_dataset = PSPnetDataset(train_lines, inputs_size, NUM_CLASSES, True)
        val_dataset = PSPnetDataset(val_lines, inputs_size, NUM_CLASSES, False)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=pspnet_dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                drop_last=True, collate_fn=pspnet_dataset_collate)

        epoch_size      = max(1, len(train_lines)//Batch_size)
        epoch_size_val  = max(1, len(val_lines)//Batch_size)

        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(Interval_Epoch,Epoch):
            fit_one_epoch(model,epoch,epoch_size,epoch_size_val,gen,gen_val,Epoch,Cuda,aux_branch)
            lr_scheduler.step()

