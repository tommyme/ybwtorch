import pandas as pd
import torch
import seaborn as sns
import os

def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

def cm2csv(epoch=0,num_cls,val_loader,model,path='',cls2id)
    conf_matrix = torch.zeros(num_cls, num_cls)
    
    for data, target in val_loader:
        output = model(data.to(device))
        conf_matrix = confusion_matrix(output, target, conf_matrix)

    df_cm = pd.DataFrame(conf_matrix.numpy(),
                         index = [i for i in list(cls2id.keys())],
                         columns = [i for i in list(cls2id.keys())])
                         
    df_cm.to_csv(os.path.join(path,'epoch_{}_conf_matrix.csv'.format(epoch)))


# how to plot?
# 列代表预测类别
# pd.read_csv('',index_col=0)
# plt.figure(figsize = (10,7))
# sns.heatmap(df_cm, annot=True, cmap="BuPu")