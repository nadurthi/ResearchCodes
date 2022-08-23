import argparse
import cv2
import open3d as o3d
import math
from models import hsm
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
from models.submodule import *
from utilshsm.eval import mkdir_p, save_pfm
from utilshsm.preprocess import get_transform
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




K1 = np.array([[     707.09,           0,      601.89],
       [          0,      707.09,      183.11],
       [          0,           0,           1]])
K2 = K1
distcoeff1 = np.array([0,0,0,0,0])
distcoeff2 = np.array([0,0,0,0,0])
imageSize = np.array([375, 1242])
Rt=np.array(  [[          1,           0,           0,     -0.5379],
               [          0,           1,  1.0808e-18,           0],
               [          0, -2.3886e-19,           1,           0],
               [          0,           0,           0,           1]])
R=Rt[0:3,0:3]
T=Rt[0:3,3]
R1, R2, P1, P2, Q, validPixROI1, validPixROI2=cv2.stereoRectify(	K1, distcoeff1, K2, distcoeff2, imageSize, R, T)

# parser = argparse.ArgumentParser(description='HSM')
# parser.add_argument('--datapath', default='./data-mbtest/',
#                     help='test data path')
# parser.add_argument('--loadmodel', default=None,
#                     help='model path')
# parser.add_argument('--outdir', default='output',
#                     help='output dir')
# parser.add_argument('--clean', type=float, default=-1,
#                     help='clean up output using entropy estimation')
# parser.add_argument('--testres', type=float, default=0.5,
#                     help='test time resolution ratio 0-x')
# parser.add_argument('--max_disp', type=float, default=-1,
#                     help='maximum disparity to search for')
# parser.add_argument('--level', type=int, default=1,
#                     help='output level of output, default is level 1 (stage 3),\
#                           can also use level 2 (stage 2) or level 3 (stage 1)')
# args = parser.parse_args()
database = "/media/na0043/misc/DATA/StereoDatasets"
datapath = '%s/stereo2015/data_scene_flow/training/'%database
# loadmodel = 'bestmodels/final-768px.tar'
loadmodel = 'bestmodels/kitti.tar'
maxdisp=max_disp = 384
testres=1
outdir='testing'

# dataloader
# from dataloader import listfiles as DA
all_left_img, all_right_img, all_left_disp,_,_,_ = lk15.dataloader('%s/stereo2015/data_scene_flow/training/'%database,typ='train') # change to trainval when finetuning on KITTI
loader_kitti15 = DA.myImageFloder(all_left_img,all_right_img,all_left_disp,mode='test', rand_scale=[0.9,2.4*1], order=0)

all_left_img, all_right_img, all_left_disp, all_right_disp = ls.dataloader('%s/Middlebury/mb-ex-training/trainingF'%database)  # mb-ex
loader_mb = DA.myImageFloder(all_left_img,all_right_img,all_left_disp,right_disparity=all_right_disp,mode='test', rand_scale=[0.225,0.6*1], rand_bright=[0.8,1.2],order=0)

loader = loader_kitti15

# f=DA.default_loader(all_left_img[0])

# test_left_img, test_right_img, _, _ = DA.dataloader(datapath)

# construct model
model = hsm(384,1,level=1)
model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if loadmodel is not None:
    pretrained_dict = torch.load(loadmodel)
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'disp' not in k}
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)
else:
    print('run with random init')
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

# dry run
multip = 48
imgL = np.zeros((1,3,24*multip,32*multip))
imgR = np.zeros((1,3,24*multip,32*multip))
imgL = Variable(torch.FloatTensor(imgL).cuda())
imgR = Variable(torch.FloatTensor(imgR).cuda())
with torch.no_grad():
    model.eval()
    pred_disp,entropy = model(imgL,imgR)


processed = get_transform()
model.eval()
inx=1

for inx in range(len(loader)):
    print(inx,len(loader))
    imgL_o = loader[inx][0]
    imgR_o = loader[inx][1]
    truedisp = loader[inx][2]
    leftfile,rightfile = loader[inx][3]
    imgL_o = (skimage.io.imread(leftfile).astype('float32'))[:,:,:3]
    imgR_o = (skimage.io.imread(rightfile).astype('float32'))[:,:,:3]
    pred_disp
    imgsize = imgL_o.shape[:2]
    
    
    
    
    if max_disp>0:
        if max_disp % 16 != 0:
            max_disp = 16 * math.floor(max_disp/16)
        max_disp = int(max_disp)
    else:
        with open(leftfile.replace('im0.png','calib.txt')) as f:
            lines = f.readlines()
            max_disp = int(int(lines[6].split('=')[-1]))

    ## change max disp
    tmpdisp = int(max_disp*testres//64*64)
    if (max_disp*testres/64*64) > tmpdisp:
        model.module.maxdisp = tmpdisp + 64
    else:
        model.module.maxdisp = tmpdisp
    if model.module.maxdisp ==64: model.module.maxdisp=128
    
    model.module.disp_reg8 =  disparityregression(model.module.maxdisp,16).cuda()
    model.module.disp_reg16 = disparityregression(model.module.maxdisp,16).cuda()
    model.module.disp_reg32 = disparityregression(model.module.maxdisp,32).cuda()
    model.module.disp_reg64 = disparityregression(model.module.maxdisp,64).cuda()
    print(model.module.maxdisp)
    
    # resize
    imgL_o = cv2.resize(imgL_o,None,fx=testres,fy=testres,interpolation=cv2.INTER_CUBIC)
    imgR_o = cv2.resize(imgR_o,None,fx=testres,fy=testres,interpolation=cv2.INTER_CUBIC)
    imgL = processed(imgL_o).numpy()
    imgR = processed(imgR_o).numpy()

    imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
    imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

    ##fast pad
    max_h = int(imgL.shape[2] // 64 * 64)
    max_w = int(imgL.shape[3] // 64 * 64)
    if max_h < imgL.shape[2]: max_h += 64
    if max_w < imgL.shape[3]: max_w += 64

    top_pad = max_h-imgL.shape[2]
    left_pad = max_w-imgL.shape[3]
    imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

    # # test
    imgL = Variable(torch.FloatTensor(imgL).cuda())
    imgR = Variable(torch.FloatTensor(imgR).cuda())
    


    # show(drawn_boxes)
    
    
    with torch.no_grad():
        torch.cuda.synchronize()
        start_time = time.time()
        pred_disp,entropy = model(imgL,imgR)
        torch.cuda.synchronize()
        ttime = (time.time() - start_time); print('time = %.2f' % (ttime*1000) )
    pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()
    
    
    top_pad   = max_h-imgL_o.shape[0]
    left_pad  = max_w-imgL_o.shape[1]
    entropy = entropy[top_pad:,:pred_disp.shape[1]-left_pad].cpu().numpy()
    pred_disp = pred_disp[top_pad:,:pred_disp.shape[1]-left_pad]
    
    # p=Image.fromarray(pred_disp.astype(int))
    
    
    # save predictions
    # idxname = test_left_img[inx].split('/')[-2]
    # if not os.path.exists('%s/%s'%(outdir,idxname)):
    #     os.makedirs('%s/%s'%(outdir,idxname))
    # idxname = '%s/disp0HSM'%(idxname)

    # resize to highres
    pred_disp = cv2.resize(pred_disp/testres,(imgsize[1],imgsize[0]),interpolation=cv2.INTER_LINEAR)

    # clip while keep inf
    invalid = np.logical_or(pred_disp == np.inf,pred_disp!=pred_disp)
    pred_disp[invalid] = 0
    
    # cv2.imshow('Disparity', np.hstack([pred_disp,truedisp]))
    cv2.imshow('Image', cv2.imread(leftfile))
    key = cv2.waitKey(0)

    
    
    idxx=np.isfinite(pred_disp)
    disperr = np.abs(pred_disp[idxx]-truedisp[idxx])
    print(np.mean(disperr),np.min(disperr))
    plt.cla()
    plt.hist(disperr,bins=np.arange(0,100,5),cumulative=True)
    plt.show()
    plt.pause(0.1)
    
    
    img = o3d.geometry.Image(pred_disp.astype(np.uint8))
    # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    # color_raw, depth_raw)
    
    
    pcd = cv2.reprojectImageTo3D(	pred_disp, Q)
    pcd[~np.isfinite(pcd)]=np.nan
    P=[]
    C=[]
    for i in range(pcd.shape[0]):
        for j in range(pcd.shape[1]):
            if np.isfinite(pcd[i,j,0]):
                P.append(pcd[i,j,:])
                C.append(imgL_o[i,j,:]/255)
    P=np.array(P)     
    C=np.array(C)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    pcd.colors = o3d.utility.Vector3dVector(C)
    o3d.visualization.draw_geometries([pcd])
     
    
    if key == 27 or key == 113:
        break
    cv2.destroyAllWindows()
    
    # cv2.reprojectImageTo3D(truedisp, Q)
    
    # np.save('%s/%s-disp.npy'% (outdir, idxname.split('/')[0]),(pred_disp))
    # np.save('%s/%s-ent.npy'% (outdir, idxname.split('/')[0]),(entropy))
    # cv2.imwrite('%s/%s-disp.png'% (outdir, idxname.split('/')[0]),pred_disp/pred_disp[~invalid].max()*255)
    # cv2.imwrite('%s/%s-ent.png'% (outdir, idxname.split('/')[0]),entropy/entropy.max()*255)

    # with open('%s/%s.pfm'% (outdir, idxname),'w') as f:
    #     save_pfm(f,pred_disp[::-1,:])
    # with open('%s/%s/timeHSM.txt'%(outdir,idxname.split('/')[0]),'w') as f:
    #      f.write(str(ttime))
        
    torch.cuda.empty_cache()

