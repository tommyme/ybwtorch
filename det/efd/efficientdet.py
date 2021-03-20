import cv2
import numpy as np
import colorsys
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from PIL import Image, ImageFont, ImageDraw
from torch.autograd import Variable
from nets.efficientdet import EfficientDetBackbone
from utils.utils import non_max_suppression, bbox_iou, decodebox, letterbox_image, efficientdet_correct_boxes
from config import classes
from draw import draw

image_sizes = [512, 640, 768, 896, 1024, 1280, 1408, 1536]


def preprocess_input(image):
    image /= 255
    mean = (0.406, 0.456, 0.485)
    std = (0.225, 0.224, 0.229)
    image -= mean
    image /= std
    return image


class EfficientDet(object):

    #---------------------------------------------------#
    #   初始化Efficientdet
    #---------------------------------------------------#
    def __init__(self, model_path, phi=0, conf=0.3, **kwargs):
        # self.__dict__.update(self._defaults)
        self.classes = classes
        self.model_path = model_path
        self.phi = phi
        self.confidence = conf
        self.init()

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#

    def init(self):
        self.net = EfficientDetBackbone(len(self.classes), self.phi).eval()

        print('Loading weights into state dict...')
        state_dict = torch.load(self.model_path)
        self.net.load_state_dict(state_dict).cuda()

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.classes), 1., 1.)
                      for x in range(len(self.classes))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):

        image_shape = np.array(np.shape(image)[0:2])

        crop_img = np.array(letterbox_image(
            image, (image_sizes[self.phi], image_sizes[self.phi])))
        photo = np.array(crop_img, dtype=np.float32)
        photo = np.transpose(preprocess_input(photo), (2, 0, 1))
        images = []
        images.append(photo)
        images = np.asarray(images)

        with torch.no_grad():
            images = torch.from_numpy(images).cuda()
            _, regression, classification, anchors = self.net(images)

            regression = decodebox(regression, anchors, images)
            detection = torch.cat([regression, classification], axis=-1)
            batch_detections = non_max_suppression(detection, len(self.classes),
                                                   conf_thres=self.confidence,
                                                   nms_thres=0.2)
        try:
            batch_detections = batch_detections[0].cpu().numpy()
        except:
            print('置信度过高，没有找到符合条件的目标')
            return image, 0, 0

        top_index = batch_detections[:, 4] > self.confidence
        top_conf = batch_detections[top_index, 4]
        top_label = np.array(batch_detections[top_index, -1], np.int32)
        top_bboxes = np.array(batch_detections[top_index, :4])
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
            top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)

        # 去掉灰条
        boxes = efficientdet_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax, np.array(
            [image_sizes[self.phi], image_sizes[self.phi]]), image_shape)

        for i, c in enumerate(top_label):
            cls = self.classes[c]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(
                bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(
                right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(cls, score)

        return image, cls, score
