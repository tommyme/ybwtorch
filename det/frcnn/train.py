from nets.frcnn import FasterRCNN
from nets.frcnn_training import Generator
from torch.autograd import Variable
from trainer import FasterRCNNTrainer
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm
import torch.optim as optim
import torch.backends.cudnn as cudnn
from config import num_classes
from torch.utils.model_zoo import load_url

def fit_ont_epoch(net,epoch,epoch_size,epoch_size_val,gen,genval,Epoch):
    train_util = FasterRCNNTrainer(net,optimizer)
    total_loss = 0
    rpn_loc_loss = 0
    rpn_cls_loss = 0
    roi_loc_loss = 0
    roi_cls_loss = 0
    val_toal_loss = 0
    with tqdm(total=epoch_size,desc=f'Train Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            start_time = time.time()
            imgs,boxes,labels = batch[0], batch[1], batch[2]

            with torch.no_grad():
                imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor)).cuda()
                boxes = [Variable(torch.from_numpy(box).type(torch.FloatTensor)).cuda() for box in boxes]
                labels = [Variable(torch.from_numpy(label).type(torch.FloatTensor)).cuda() for label in labels]
            losses = train_util.train_step(imgs, boxes, labels, 1)
            rpn_loc, rpn_cls, roi_loc, roi_cls, total = losses
            total_loss += total
            rpn_loc_loss += rpn_loc
            rpn_cls_loss += rpn_cls
            roi_loc_loss += roi_loc
            roi_cls_loss += roi_cls

            waste_time = time.time() - start_time
            
            pbar.set_postfix(**{'total_loss'      : float(total_loss/(iteration+1)), 
                                'rpn_loc_loss'    : float(rpn_loc_loss/(iteration+1)), 
                                'rpn_cls_loss'    : float(rpn_cls_loss/(iteration+1)),
                                'roi_loc_loss'    : float(roi_loc_loss/(iteration+1)),
                                'roi_cls_loss'    : float(roi_cls_loss/(iteration+1))})
            pbar.update(1)
            

    with tqdm(total=epoch_size_val, desc=f'Val Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            imgs,boxes,labels = batch[0], batch[1], batch[2]
            with torch.no_grad():
                imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor)).cuda()
                boxes = Variable(torch.from_numpy(boxes).type(torch.FloatTensor)).cuda()
                labels = Variable(torch.from_numpy(labels).type(torch.FloatTensor)).cuda()

                train_util.optimizer.zero_grad()
                losses = train_util.forward(imgs, boxes, labels, 1)
                _,_,_,_, val_total = losses
                val_toal_loss += val_total
                
            pbar.set_postfix(**{'total_loss': val_toal_loss / (iteration + 1)})
            pbar.update(1)
            
    print('\nEpoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_toal_loss/(epoch_size_val+1)))

    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), './Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_toal_loss/(epoch_size_val+1)))

#----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--lr", type=float, help="learning_rate", default=1e-3)
    parser.add_argument("-b", "--backbone", type=str, help="frozen_epoches", default='resnet50', choices=['resnet50','vgg'])
    parser.add_argument("-f", "--freeze", type=int, help="frozen_epoches", default=5)
    parser.add_argument("-u", "--unfreeze", type=int, help="free_epoches", default=5)
    # parser.add_argument('-v', "--version", type=int, help='you want to use efficientdet-dX?', default=0)
    # parser.add_argument('-b', "--batch_size", type=int, help='just batch_size', default=64)
    parser.add_argument('-p', "--pre_model", type=str, help='just pretrained_model_path default = \'\'', default="")
    parser.add_argument('-s', '--val_split', type=float, help='验证集的比例', default=0.1)
    
    args = parser.parse_args()

    # 参数初始化
    annotation_path = '2007_train.txt'
    EPOCH_LENGTH = 2000
    IMAGE_SHAPE = [600,600,3]
    model = FasterRCNN(num_classes,backbone=args.backbone).cuda()
    model_urls = {'resnet50':'https://github.com/you-bowen/tutorical_myDL/releases/download/1.0/frcnn_resnet50.pth',
                  'vgg':'none'}
    #-------------------------------------------#
    #   权值文件的下载请看README
    #-------------------------------------------#
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(args.pre_model) if args.pre_model else load_url(model_urls[args.backbone], map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v) and k[-19:] != 'num_batches_tracked'}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(model.state_dict()['extractor.4.0.bn1.num_batches_tracked'])
    print('Finished!')

    cudnn.benchmark = True

    # 0.1用于验证，0.9用于训练
    val_split = args.val_split
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    


    lr = args.lr
    Init_Epoch = int(args.pre_model.split('/')[-1].split('-')[0][5:]) if args.pre_model else 0
    Freeze_Epoch = args.freeze
    Unfreeze_Epoch = args.unfreeze

    optimizer = optim.Adam(model.parameters(),lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)

    gen = Generator(lines[:num_train],(IMAGE_SHAPE[0],IMAGE_SHAPE[1])).generate()
    gen_val = Generator(lines[num_train:],(IMAGE_SHAPE[0],IMAGE_SHAPE[1])).generate()
    
    epoch_size = EPOCH_LENGTH
    epoch_size_val = int(EPOCH_LENGTH/10)
    
    for param in model.extractor.parameters():
        param.requires_grad = False
    print('freezed !')
    for epoch in range(Init_Epoch,Freeze_Epoch+Unfreeze_Epoch):
        if epoch == Freeze_Epoch:
            lr = lr / 10
            for param in model.extractor.parameters():
                param.requires_grad = True
            print('unfreezed !')
        fit_ont_epoch(model,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch)
        lr_scheduler.step()
