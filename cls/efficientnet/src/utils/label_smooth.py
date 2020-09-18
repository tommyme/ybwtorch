#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import utils.config as cfg

class LabelSmoothSoftmaxCE(nn.Module):
    def __init__(self,
                 lb_pos=0.9,
                 lb_neg=0.005,
                 reduction='mean',
                 lb_ignore=255,
                 ):
        super(LabelSmoothSoftmaxCE, self).__init__()
        self.lb_pos = lb_pos
        self.lb_neg = lb_neg
        self.reduction = reduction
        self.lb_ignore = lb_ignore
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, logits, label):
        if cfg.use_sample_weight:
            weights = torch.Tensor([cfg.sample_weight[int(i)] for i in label]).cuda()
        logs = self.log_softmax(logits)
        ignore = label.data.cpu() == self.lb_ignore
        n_valid = (ignore == 0).sum()
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        label = self.lb_pos * lb_one_hot + self.lb_neg * (1-lb_one_hot)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        label[[a, torch.arange(label.size(1)), *b]] = 0
        if cfg.use_sample_weight:
            loss = -torch.sum(torch.sum(logs*label, dim=1)*weights) / n_valid
        else:
            loss = -torch.sum(torch.sum(logs*label, dim=1)) / n_valid

        return loss


if __name__ == '__main__':
    torch.manual_seed(15)
    criteria = LabelSmoothSoftmaxCE(lb_pos=0.9, lb_neg=5e-3)
    net1 = nn.Sequential(
        nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
    )
    net1.cuda()
    net1.train()
    net2 = nn.Sequential(
        nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
    )
    net2.cuda()
    net2.train()

    with torch.no_grad():
        inten = torch.randn(2, 3, 5, 5).cuda()
        lbs = torch.randint(0, 3, [2, 5, 5]).cuda()
        lbs[1, 3, 4] = 255
        lbs[1, 2, 3] = 255
        print(lbs)

    import torch.nn.functional as F
    logits1 = net1(inten)
    logits1 = F.interpolate(logits1, inten.size()[2:], mode='bilinear')
    logits2 = net2(inten)
    logits2 = F.interpolate(logits2, inten.size()[2:], mode='bilinear')

    #  loss1 = criteria1(logits1, lbs)
    loss = criteria(logits1, lbs)
    #  print(loss.detach().cpu())
    loss.backward()