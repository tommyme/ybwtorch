from nets.ssd import get_ssd
from nets.ssd_training import Generator,MultiBoxLoss
from torch.utils.data import DataLoader
from utils.dataloader import ssd_dataset_collate, SSDDataset
from utils.config import Config
from torchsummary import summary
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import config
import argparse
from torch.utils.model_zoo import load_url
from tqdm import tqdm
import urllib
def adjust_learning_rate(optimizer, lr, gamma, step):
    lr = lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_one_epoch(lr, epoch, gen, net, optimizer, criterion, epoch_size, Epoch):

    if epoch%2==0:
        adjust_learning_rate(optimizer,lr,0.9,epoch)
    loc_loss = 0
    conf_loss = 0
    with tqdm(total=epoch_size,desc=f'Train Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets]

            # 前向传播
            out = net(images)
            # 清零梯度
            optimizer.zero_grad()
            # 计算loss
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            # 反向传播
            loss.backward()
            optimizer.step()
            # 加上
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()
            
            pbar.set_postfix(**{'Loc_Loss':   loc_loss/(iteration+1),
                                'Conf_Loss':  conf_loss/(iteration+1)})
            pbar.update(1)
            
    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), './Epoch%d-Loc%.4f-Conf%.4f.pth'%((epoch+1),loc_loss/(iteration+1),conf_loss/(iteration+1)))
    



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--lr", type=float, help="learning_rate", default=1e-3)
    parser.add_argument("-f", "--freeze", type=int, help="frozen_epoches", default=5)
    parser.add_argument("-u", "--unfreeze", type=int, help="free_epoches", default=5)
    parser.add_argument('-b', "--batch_size", type=int, help='just batch_size', default=32)
    parser.add_argument('-p', "--pre_model", type=str, help='just pretrained_model_path default = \'\'', default='')

    args = parser.parse_args()
    
    # 参数初始化
    
    Init_Epoch = int(args.pre_model.split('/')[-1].split('-')[0][5:]) if args.pre_model else 0
    model = get_ssd("train",config.num_classes)
    
    # 读取权重

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    # 旧版pytorch有哈希验证，这里咱也不爱改名字，就这样吧
    urllib.request.urlretrieve('https://github.com/you-bowen/tutorical_myDL/releases/download/1.0/ssd_weights.pth', "ssd_weights.pth")
    pretrained_dict = torch.load('ssd_weights.pth', map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('weights loaded')
    
    # 网络进入cuda
    
    net = model.train()
    net = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    net = net.cuda()
    
    # 拆分训练集
    
    with open('2007_train.txt') as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_train = len(lines)

    train_dataset = SSDDataset(lines[:num_train], (Config["min_dim"], Config["min_dim"]))
    gen = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                            drop_last=True, collate_fn=ssd_dataset_collate)

    criterion = MultiBoxLoss(config.num_classes, 0.5, True, 0, True, 3, 0.5,
                             False, True)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    epoch_size = num_train // args.batch_size



    for param in model.vgg.parameters():
        param.requires_grad = False

    for epoch in range(Init_Epoch, args.freeze + args.unfreeze):
        # 解冻逻辑
        if epoch == args.freeze:
            print('unfreezing')
            args.lr /= 10
            for param in model.vgg.parameters():
                param.requires_grad = True

        train_one_epoch(args.lr, epoch, gen, net, optimizer, criterion, epoch_size, args.freeze + args.unfreeze)



