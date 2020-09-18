import utils.config as cfg
from nets.efficientnet_pytorch import EfficientNet, cbam_EfficientNet
from nets.resnet import resnet50, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2

def get_net(net_cfg):
    
    resnets = {'resnet50':resnet50, 'resnext50_32x4d':resnext50_32x4d, 'resnext101_32x8d':resnext101_32x8d, 'wide_resnet50_2':wide_resnet50_2, 'wide_resnet101_2':wide_resnet101_2}
    
    if net_cfg['model'][:12] == 'efficientnet': # efficientnet-b4 or efficientnet-cbam-2-b4
        v = net_cfg['model'].split('-')[-1][-1]
        cbam = net_cfg['model'].split('-')[-2]
        return cbam_EfficientNet.from_pretrained('efficientnet-b{}'.format(v),weights_path=net_cfg['pre_model'],num_classes=cfg.num_classes,cbam=cfg.cbam)
    
    elif net_cfg['model'] in ['resnet50', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2']:
        return resnets[net_cfg['model']](pretrained=True,pre_model=net_cfg['pre_model'])
