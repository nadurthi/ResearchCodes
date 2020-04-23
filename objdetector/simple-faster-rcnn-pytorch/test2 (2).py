#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 17:28:59 2020

@author: nagnanamus
"""

import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
import os
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at
# %matplotlib inline
import cupy as cp
import numpy as np
order = cp.arange(3, dtype=np.int32)


img = read_image('misc/demo.jpg')
print(img.shape)
img = t.from_numpy(img)[None]
img.shape

faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()


trainer.load('fasterrcnn_12211511_0.701052458187_torchvision_pretrain.pth.701052458187')
opt.caffe_pretrain=False # this model was trained from caffe-pretrained model
_bboxes, _labels, _scores = trainer.faster_rcnn.predict(img,visualize=True)
vis_bbox(at.tonumpy(img[0]),
         at.tonumpy(_bboxes[0]),
         at.tonumpy(_labels[0]).reshape(-1),
         at.tonumpy(_scores[0]).reshape(-1))

plt.show()

