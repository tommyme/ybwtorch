#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from efficientdet import EfficientDet
from PIL import Image
import argparse
import os
import matplotlib.pyplot as plt
import random

def show_imgs(imgs, labels, scores, cols=5):
    '''
    参数全是列表。
    '''
    num = len(imgs)
    fig, ax = plt.subplots((num-1)//cols+1,cols)
    plt.gcf().set_size_inches(15,7) # 改变参数以调节间距
    for i in range(num):
        plt.subplot(num//cols+1,cols,i+1)
        plt.imshow(imgs[i])
        plt.title(labels[i]+' '+str(scores[i]), fontsize=16)
        plt.xticks([])
        plt.yticks([])
    fig.savefig('temp.png', quanlity=100, dpi=100)


parser = argparse.ArgumentParser(description="...")

parser.add_argument("-p", "--model_path", type=str, help="frozen_epoches")
parser.add_argument("-c", "--conf", type=float, help="cinfidence", default=0.3)
parser.add_argument('-v', "--version", type=int, help='you want to use efficientdet-dX?', default=0)
parser.add_argument('-g', "--cuda", type=bool, help='do you have a gpu?', default=True)
parser.add_argument('-n', "--num2show", type=int, help='num img 2 show', default=1)
parser.add_argument('-r', "--root", type=str, help='root dir filled with *.jpg')
parser.add_argument('-i', "--filename", type=str, help='filename', default='')

args = parser.parse_args()
    
efficientdet = EfficientDet(args.model_path, args.version, args.conf, args.cuda)

if args.num2show == 1:
    image = Image.open(os.path.join(args.root,args.filename))
    res, cls, score = efficientdet.detect_image(image)
    print(cls, score)
    # r_image.show()
    
else:
    print('结果将会保存到temp.png')
    files = os.listdir(args.root)
    idx = [int(len(os.listdir(args.root))*random.random()) for i in range(args.num2show)]
    imgs = [Image.open(os.path.join(args.root,files[id])) for id in idx]
    ress, clss, scores = [], [], []
    print(len(imgs))
    for img in imgs:
        res, cls, score = efficientdet.detect_image(img)
        ress.append(res)
        clss.append(cls)
        scores.append(score)
    show_imgs(ress, clss, scores)
    
