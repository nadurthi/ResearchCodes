import argparse

import math
import numpy as np
import os
import matplotlib.pyplot as plt
import pdb
import skimage.io
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import time

#cudnn.benchmark = True
cudnn.benchmark = False
from dataloader import listfiles as ls
from dataloader import listsceneflow as lt
from dataloader import KITTIloader2015 as lk15
from dataloader import KITTIloader2012 as lk12
from dataloader import MiddleburyLoader as DA
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
from torchvision.ops import nms
import cv2

# load a model pre-trained on COCO
fasterRcnnmodel = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
fasterRcnnmodel.cuda()
fasterRcnnmodel.eval()

fasterRcnnmodel_fpn_v2 = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
fasterRcnnmodel_fpn_v2.cuda()
fasterRcnnmodel_fpn_v2.eval()

fasterRcnnmodel_mblnet_highres = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
fasterRcnnmodel_mblnet_highres.cuda()
fasterRcnnmodel_mblnet_highres.eval()

fasterRcnnmodel_mblnet_lowres = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
fasterRcnnmodel_mblnet_lowres.cuda()
fasterRcnnmodel_mblnet_lowres.eval()


ssdvgg16 = torchvision.models.detection.ssd300_vgg16(weights=
torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT)
ssdvgg16.cuda()
ssdvgg16.eval()

ssdlite = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
ssdlite.cuda()
ssdlite.eval()

yolov5s = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolov5s.cuda()
yolov5s.eval()

database = "/media/na0043/misc/DATA/StereoDatasets"
datapath = '%s/stereo2015/data_scene_flow/training/'%database


# dataloader
# from dataloader import listfiles as DA
all_left_img, all_right_img, all_left_disp,_,_,_ = lk15.dataloader('%s/stereo2015/data_scene_flow/training/'%database,typ='train') # change to trainval when finetuning on KITTI
loader_kitti15 = DA.myImageFloder(all_left_img,all_right_img,all_left_disp,mode='test', rand_scale=[0.9,2.4*1], order=0)

all_left_img, all_right_img, all_left_disp, all_right_disp = ls.dataloader('%s/Middlebury/mb-ex-training/trainingF'%database)  # mb-ex
loader_mb = DA.myImageFloder(all_left_img,all_right_img,all_left_disp,right_disparity=all_right_disp,mode='test', rand_scale=[0.225,0.6*1], rand_bright=[0.8,1.2],order=0)

loader = loader_kitti15
inx=1

fasterrcnn_t = FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms()
ssd_t = torchvision.models.detection.SSD300_VGG16_Weights.COCO_V1.transforms()
ssdlite_t = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1.transforms()

for inx in range(len(loader)):
    print(inx,len(loader))
    imgL_o = loader[inx][0]
    imgR_o = loader[inx][1]
    truedisp = loader[inx][2]
    leftfile,rightfile = loader[inx][3]

    
    img = read_image(leftfile).cuda()
    img_t = fasterrcnn_t(img)
    
    
    
    st=time.time()
    predictions_fastfpnv2 = fasterRcnnmodel_fpn_v2([img_t])
    et=time.time()
    print("fasterRcnnmodel_fpn_v2 detection = ",et-st)
    torch.cuda.empty_cache()
    
    st=time.time()
    predictions_fstmblnet_high = fasterRcnnmodel_mblnet_highres([img_t])
    et=time.time()
    print("fasterRcnnmodel_mblnet_highres detection = ",et-st)
    torch.cuda.empty_cache()
    
    st=time.time()
    prediction_fstsmblnet_low = fasterRcnnmodel_mblnet_lowres([img_t])
    et=time.time()
    print("fasterRcnnmodel_mblnet_lowres detection = ",et-st)
    
    st=time.time()
    predictions_fst = fasterRcnnmodel([img_t])
    et=time.time()
    print("fasterRcnnmodel detection = ",et-st)
    torch.cuda.empty_cache()
    
    st=time.time()
    predictions_ssd = ssdvgg16([img_t])
    et=time.time()
    print("ssdvgg16 detection = ",et-st)
    torch.cuda.empty_cache()
    
    st=time.time()
    predictions_ssdlite = ssdlite([img_t])
    et=time.time()
    print("ssdlite detection = ",et-st)
    torch.cuda.empty_cache()
    
    
    st=time.time()
    predictions_yolo5s = yolov5s([leftfile])
    et=time.time()
    print("yolov5s detection = ",et-st)
    torch.cuda.empty_cache()
    pred_yolo5s=[]
    for dd in predictions_yolo5s.pandas().xyxy:
        pred_yolo5s.append({'boxes':dd[['xmin','ymin','xmax','ymax']].values,
                            'scores':dd['confidence'].values,
                            'labels':dd['name'].values})
    
    
    
    
    pred = predictions_fastfpnv2
    
    # scores = pred[0]['scores']
    
    scores=np.asarray(pred[0]['scores'].cpu().detach())
    boxes = np.asarray(pred[0]['boxes'].cpu().detach())
    labels = np.asarray(pred[0]['labels'].cpu().detach())
    
    idd2=nms(pred[0]['boxes'],pred[0]['scores'],0.1).cpu().detach().numpy()
    idd = scores[idd2]>0.15
    
    scores_f = scores[idd2][idd]
    boxes_f = boxes[idd2][idd].astype(int)
    labels_f = labels[idd2][idd]
    
    cv2imgg = np.asarray(T.ToPILImage()(img))
    
    # drawn_boxes = draw_bounding_boxes(img, pred[0]['boxes'][idd2][idd], colors="red")
    for i in range(len(scores_f)):    
        cv2.rectangle(cv2imgg,(boxes_f[i][0],boxes_f[i][1]),(boxes_f[i][2],boxes_f[i][3]),(0,255,0),2)
        cv2.putText(cv2imgg,str(labels_f[i]),(boxes_f[i][0],boxes_f[i][1]-10),0,0.3,(0,255,0))
    
    cv2.imshow('boxes',cv2imgg )
    key = cv2.waitKey(0)

    if key == 27 or key == 113:
        break
    cv2.destroyAllWindows()
    
           
    

