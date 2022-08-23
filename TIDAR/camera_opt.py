#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 14:36:56 2022

@author: na0043
"""

import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import pickle as pkl
from utils.parconsumer import ParallelConsumer
import multiprocessing as mp
import queue
import pandas as pd

plt.close("all")
inch2m = 0.0254
microm2m = 1e-6
cm2m = 1e-2

# The Imaging source

# focal lengths in m
flens = np.array([4,6,8,12,16,25,35,50,75])*1e-3

# sensors width and height (square sensor)
sensors=[]
sensors.append({'id':1,'pxs': 3.45 * microm2m,'npx':4096,'npy':2160, 'w': 4096*3.45 * microm2m, 'h': 2160*3.45 * microm2m, 'link': 'https://www.theimagingsource.com/products/industrial-cameras/usb-3.1-monochrome/dmk38ux267/'  })
sensors.append({'id':2,'pxs': 3.45 * microm2m,'npx':4096,'npy':3000, 'w': 4096*3.45 * microm2m, 'h': 3000*3.45 * microm2m, 'link': 'https://www.theimagingsource.com/products/industrial-cameras/usb-3.1-monochrome/dmk38ux304/'  })
sensors.append({'id':3,'pxs': 2.74 * microm2m,'npx':5320,'npy':3032, 'w': 5320*2.74 * microm2m, 'h': 3032*2.74 * microm2m,  'link': 'https://www.theimagingsource.com/products/industrial-cameras/usb-3.1-monochrome/dmk38ux542/'})
sensors.append({'id':4,'pxs': 2.74 * microm2m,'npx':4504,'npy':4504, 'w': 4504*2.74 * microm2m, 'h': 4504*2.74 * microm2m,  'link': 'https://www.theimagingsource.com/products/industrial-cameras/usb-3.1-monochrome/dmk38ux541/'})
sensors.append({'id':5,'pxs': 2.74 * microm2m,'npx':5320,'npy':4600, 'w': 5320*2.74 * microm2m, 'h': 4600*2.74 * microm2m,  'link': 'https://www.theimagingsource.com/products/industrial-cameras/usb-3.1-monochrome/dmk38ux540/'})


sensors.append({'id':6,'pxs': 6.9 * microm2m,'npx':720,'npy':540, 'w': 720*6.9 * microm2m, 'h': 540*6.9 * microm2m,  'link': 'https://www.theimagingsource.com/products/industrial-cameras/usb-3.1-monochrome/dmk37aux287/'})
sensors.append({'id':7,'pxs': 3.45 * microm2m,'npx':1440,'npy':1080, 'w': 1440*3.45 * microm2m, 'h': 1080*3.45 * microm2m,  'link': 'https://www.theimagingsource.com/products/industrial-cameras/usb-3.1-monochrome/dmk37aux273/'})
sensors.append({'id':8,'pxs': 2.9 * microm2m,'npx':1920,'npy':1080, 'w': 1920*2.9 * microm2m, 'h': 1080*2.9 * microm2m,  'link': 'https://www.theimagingsource.com/products/industrial-cameras/usb-3.1-monochrome/dmk37aux290/'})
sensors.append({'id':9,'pxs': 3.45 * microm2m,'npx':2048,'npy':1536, 'w': 2048*3.45 * microm2m, 'h': 1536*3.45 * microm2m,  'link': 'https://www.theimagingsource.com/products/industrial-cameras/usb-3.1-monochrome/dmk37aux252/'})
sensors.append({'id':10,'pxs': 3.45 * microm2m,'npx':2448,'npy':2048, 'w': 2448*3.45 * microm2m, 'h': 2048*3.45 * microm2m,  'link': 'https://www.theimagingsource.com/products/industrial-cameras/usb-3.1-monochrome/dmk37aux250//'})
sensors.append({'id':11,'pxs': 2.4 * microm2m,'npx':3072,'npy':2048, 'w': 3072*2.4 * microm2m, 'h': 2048*2.4 * microm2m,  'link': 'https://www.theimagingsource.com/products/industrial-cameras/usb-3.1-monochrome/dmk37aux178/'})
sensors.append({'id':12,'pxs': 1.85 * microm2m,'npx':4000,'npy':3000, 'w': 4000*1.85 * microm2m, 'h': 3000*1.85 * microm2m,  'link': 'https://www.theimagingsource.com/products/industrial-cameras/usb-3.1-monochrome/dmk37aux226/'})



def flagPointsInFrustum(X,center,f,w,h):
    # center is camera sensor center
    # f is focal length, w and h are width and height
    XX = X - np.array(center)
    XX[:,0]=f*XX[:,0]/XX[:,2]
    XX[:,1]=f*XX[:,1]/XX[:,2]
    XX[:,2]=f
    return (XX[:,0]<=w/2) & (XX[:,0]>=-w/2) & (XX[:,1]<=h/2) & (XX[:,1]>=-h/2)


# camera Z axis faces forward, x axis to the right

# constraints 
max_baseline = 1 # meters
number_cameras = 2

human_height = 1.5
human_width = 0.5

human_height_pixels = 70
human_width_pixels = 20

epsilon_d = 1 # disparity in number of pixels
max_epsilon_z = 2 #100*cm2m  # max error possible in depth estimate

# Sensing volume FOV (in meters)
FOV_height=50
FOV_width=200
FOV_depth=100 
# dx= 10*cm2m 

# sensor is center of face of sensing volume
Xvoxels,Yvoxels,Zvoxels = np.meshgrid(np.arange(-FOV_width/2,FOV_width/2,human_width),
                     np.arange(-FOV_height/2,FOV_height/2,human_height),
                     np.arange(0,FOV_depth,human_width)) 

voxels_centers = np.vstack([Xvoxels.reshape(-1),Yvoxels.reshape(-1),Zvoxels.reshape(-1)]).T
# human target is placed at the center of the voxel
# first lower face of cuboid starting from front left corner
hh0= np.vstack( [voxels_centers[:,0]-human_width/2,voxels_centers[:,1]+human_height/2,voxels_centers[:,2]-human_width/2]).T
hh1= hh0.copy()
hh1[:,0]=hh1[:,0]+human_width
hh2=hh1.copy()
hh2[:,2]=hh2[:,2]+human_width
hh3=hh2.copy()
hh3[:,0]=hh3[:,0]-human_width

human_voxel_corner = np.hstack([ hh0, hh1, hh2, hh3 ])
hh = human_voxel_corner.copy()
hh[:,1]=hh[:,1]+human_height
human_voxel_corners=np.hstack([human_voxel_corner,hh])

#%%
###############  2 Cameras ############################

# Sols2cam=[{'hullstereovisibility':0,'hullLowErr':0,'hulldetection':0,'f':0,'sensID':0,'b':0}]
Sols2cam=[]
for b in np.arange(1*cm2m,max_baseline,1*cm2m):
    cameras_centers = [np.array([-b/2,0,0]),np.array([b/2,0,0])]
    # iterate over cameras
    for sens in sensors:
        for f in flens:
            print(b,sens['id'],f)
            # cost of points visible by atlest 2 cameras
            camera_insideFlags=[]
            for i in range(len(cameras_centers)):
                center = cameras_centers[i]
                cornerflgs=[]
                for j in range(0,24,3):
                    flg=flagPointsInFrustum(human_voxel_corners[:,j:j+3],center,f,sens['w'],sens['h'])
                    cornerflgs.append(flg)
                
                cornerflgs=np.vstack(cornerflgs).T
                # all corners inside the frustum of camera
                camera_insideFlags.append(np.all(cornerflgs,axis=1))
                
            camera_insideFlags=np.vstack(camera_insideFlags).T
            
            stereo_visibility = np.all(camera_insideFlags,axis=1)
            stereo_voxels_centers = voxels_centers[stereo_visibility,:]
            pts=stereo_voxels_centers[:,[0,2]]
            hullstereovisibility = ConvexHull(pts)
            hullstereovisibilitypts = pts[hullstereovisibility.vertices,:]
            cost_FOV = hullstereovisibility.volume
            cost_stereo_percent = 100*stereo_voxels_centers.shape[0]/voxels_centers.shape[0]
            
            
            # caculate depth error in m (remember to convert disparity error to m)
            total_max_depth_error = ( stereo_voxels_centers[:,2]**2 )*(epsilon_d * sens['pxs']/(b*f))
            stereo_lowErr_voxels_centers = stereo_voxels_centers[total_max_depth_error<=max_epsilon_z,:]
            pts=stereo_lowErr_voxels_centers[:,[0,2]]
            if len(stereo_lowErr_voxels_centers)>0:
                try:
                    hullLowErr = ConvexHull(pts)
                    hullLowErrpts = pts[hullLowErr.vertices,:]
                    cost_LowErr = hullLowErr.volume
                except:
                    hullLowErr = None
                    cost_LowErr = 0
                    hullLowErrpts=[]
            else:
                cost_LowErr = 0
                hullLowErrpts=[]
                
            # number of pixels in each camera, assuming the target is facing camera
            pixel_cost_flg_per_camera=[]
            for i in range(len(cameras_centers)):
                PP=np.ones(voxels_centers.shape[0],dtype=bool)
                PP[camera_insideFlags[:,i]==False]=False
                ww = f*human_width/(voxels_centers[camera_insideFlags[:,i]==True,2]-human_width/2)
                hh = f*human_height/(voxels_centers[camera_insideFlags[:,i]==True,2]-human_height/2)
                npixW = ww/sens['pxs']
                npixH = hh/sens['pxs']
                PP[camera_insideFlags[:,i]==True]=(npixH>=human_height_pixels) & (npixW>=human_width_pixels)
                pixel_cost_flg_per_camera.append( PP  )
            
            pixel_cost_flg_per_camera = np.vstack(pixel_cost_flg_per_camera).T
            
            pixel_detection_cost_flg = np.any(pixel_cost_flg_per_camera,axis=1)
            pts = voxels_centers[pixel_detection_cost_flg,:]
            pts = pts[:,[0,2]]
            hulldetection = ConvexHull(pts) 
            
            hulldetectionpts = pts[hulldetection.vertices,:]
            cost_detection = hulldetection.volume
            
            
            ff = "TIDAR/2Cam/2Cam b:%f, f:%f, sensID:%d"%(b,f,sens['id'])
            
            # fig=plt.figure()
            # plt.plot(hullstereovisibilitypts[:,0], hullstereovisibilitypts[:,1], 'b^-', lw=2,label='stereo-visible')
            # if len(hullLowErrpts)>0:
            #     plt.plot(hullLowErrpts[:,0], hullLowErrpts[:,1], 'go-', lw=2,label='DepthErr')
            # plt.plot(hulldetectionpts[:,0], hulldetectionpts[:,1], 'rs-', lw=2,label='detection')
            # plt.legend()
            # plt.title("b:%f, f:%f, pxs:%f, npx:%d, npy:%d"%(b,f,sens['pxs'],sens['w']/sens['pxs'],sens['h']/sens['pxs']))
            # plt.xlim((-FOV_width/2-10,FOV_width/2+10))
            # plt.ylim((0,FOV_depth))
            # plt.grid()
            # plt.show()
            
            
            # plt.savefig("b:%f, f:%f, pxs:%f, npx:%d, npy:%d"%(b,f,sens['pxs'],sens['w']/sens['pxs'],sens['h']/sens['pxs']),format='png')
            # plt.close(fig)
            
            # with open(ff+'.pkl','wb') as F:
            #     pkl.dump({'hullstereovisibility':hullstereovisibility,
            #                      'hullLowErr':hullLowErr,'hulldetection':hulldetection,
            #                      'f':f,'sensID':sens['id'],'b':b,
            #                      'camera_insideFlags':camera_insideFlags,'cost_FOV':cost_FOV,
            #                      'total_max_depth_error':total_max_depth_error,
            #                      'cost_LowErr':cost_LowErr,'pixel_cost_flg_per_camera':pixel_cost_flg_per_camera,
            #                      'cost_detection':cost_detection},F)
                
            Sols2cam.append({#'hullstereovisibility':hullstereovisibility,
                              #'hullLowErr':hullLowErr,'hulldetection':hulldetection,
                              'f':f,'sensID':sens['id'],'b':b,
                              #'camera_insideFlags':camera_insideFlags,
                              'cost_FOV':cost_FOV,
                              # 'total_max_depth_error':total_max_depth_error,
                              'cost_LowErr':cost_LowErr,
                              #'pixel_cost_flg_per_camera':pixel_cost_flg_per_camera,
                              'cost_detection':cost_detection,
                              'hullstereovisibilitypts':hullstereovisibilitypts,
                              'hulldetectionpts':hulldetectionpts,
                              'hullLowErrpts':hullLowErrpts})
            

#%%

def getmetrics(sens,f,cameras_centers):
    # cost of points visible by atlest 2 cameras
    camera_insideFlags=[]
    for i in range(len(cameras_centers)):
        center = cameras_centers[i]
        cornerflgs=[]
        for j in range(0,24,3):
            flg=flagPointsInFrustum(human_voxel_corners[:,j:j+3],center,f,sens['w'],sens['h'])
            cornerflgs.append(flg)
        
        cornerflgs=np.vstack(cornerflgs).T
        # all corners inside the frustum of camera
        camera_insideFlags.append(np.all(cornerflgs,axis=1))
        
    camera_insideFlags=np.vstack(camera_insideFlags).T
    
    stereo_visibility = np.sum(camera_insideFlags,axis=1)>=2
    stereo_voxels_centers = voxels_centers[stereo_visibility,:]
    pts=stereo_voxels_centers[:,[0,2]]
    hullstereovisibility = ConvexHull(pts)
    hullstereovisibilitypts = pts[hullstereovisibility.vertices,:]
    cost_FOV = hullstereovisibility.volume
    cost_stereo_percent = 100*stereo_voxels_centers.shape[0]/voxels_centers.shape[0]
    
    
    # caculate depth error in m (remember to convert disparity error to m)
    B=np.zeros((len(stereo_visibility),len(cameras_centers)))
    svflgs = camera_insideFlags[stereo_visibility,:]
    for i in range(len(cameras_centers)):
        B[:,i] = cameras_centers[i][0]

    bb=np.zeros(stereo_voxels_centers.shape[0])
    for i in range(svflgs.shape[0]):
        BB=B[i,svflgs[i]]
        bb[i]=np.max(BB)-np.min(BB)
    total_max_depth_error = ( stereo_voxels_centers[:,2]**2 )*(epsilon_d * sens['pxs']/(bb*f))
    stereo_lowErr_voxels_centers = stereo_voxels_centers[total_max_depth_error<=max_epsilon_z,:]
    pts=stereo_lowErr_voxels_centers[:,[0,2]]
    if len(stereo_lowErr_voxels_centers)>0:
        try:
            hullLowErr = ConvexHull(pts)
            hullLowErrpts = pts[hullLowErr.vertices,:]
            cost_LowErr = hullLowErr.volume
        except:
            hullLowErr = None
            cost_LowErr = 0
            hullLowErrpts=[]
    else:
        cost_LowErr = 0
        hullLowErrpts=[]
        hullLowErr = None
        
    # number of pixels in each camera, assuming the target is facing camera
    pixel_cost_flg_per_camera=[]
    for i in range(len(cameras_centers)):
        PP=np.ones(voxels_centers.shape[0],dtype=bool)
        PP[camera_insideFlags[:,i]==False]=False
        ww = f*human_width/(voxels_centers[camera_insideFlags[:,i]==True,2]-human_width/2)
        hh = f*human_height/(voxels_centers[camera_insideFlags[:,i]==True,2]-human_height/2)
        npixW = ww/sens['pxs']
        npixH = hh/sens['pxs']
        PP[camera_insideFlags[:,i]==True]=(npixH>=human_height_pixels) & (npixW>=human_width_pixels)
        pixel_cost_flg_per_camera.append( PP  )
    
    pixel_cost_flg_per_camera = np.vstack(pixel_cost_flg_per_camera).T
    
    pixel_detection_cost_flg = np.any(pixel_cost_flg_per_camera,axis=1)
    pts = voxels_centers[pixel_detection_cost_flg,:]
    pts = pts[:,[0,2]]
    hulldetection = ConvexHull(pts) 
    
    hulldetectionpts = pts[hulldetection.vertices,:]
    cost_detection = hulldetection.volume
    
    print("done")
    return {'hullstereovisibility':hullstereovisibility,
                          'hullLowErr':hullLowErr,'hulldetection':hulldetection,
                          'f':f,'sensID':sens['id'],'cameras_centers':cameras_centers,
                          'camera_insideFlags':camera_insideFlags,'cost_FOV':cost_FOV,
                          'total_max_depth_error':total_max_depth_error,
                          'cost_LowErr':cost_LowErr,'pixel_cost_flg_per_camera':pixel_cost_flg_per_camera,
                          'cost_detection':cost_detection,
                          'hullstereovisibilitypts':hullstereovisibilitypts,
                          'hulldetectionpts':hulldetectionpts,
                          'hullLowErrpts':hullLowErrpts}
    
    # ff = "3Cam/3Cam b:%f, f:%f, sensID:%d"%(b,f,sens['id'])
    
    # fig=plt.figure()
    # plt.plot(hullstereovisibilitypts[:,0], hullstereovisibilitypts[:,1], 'b^-', lw=2,label='stereo-visible')
    # if len(hullLowErrpts)>0:
    #     plt.plot(hullLowErrpts[:,0], hullLowErrpts[:,1], 'go-', lw=2,label='DepthErr')
    # plt.plot(hulldetectionpts[:,0], hulldetectionpts[:,1], 'rs-', lw=2,label='detection')
    # plt.legend()
    # plt.title("b:%f, f:%f, pxs:%f, npx:%d, npy:%d"%(b,f,sens['pxs'],sens['w']/sens['pxs'],sens['h']/sens['pxs']))
    # plt.xlim((-FOV_width/2-10,FOV_width/2+10))
    # plt.ylim((0,FOV_depth))
    # plt.grid()
    # plt.show()
    
    
    # plt.savefig(ff,format='png')
    # plt.close(fig)
    
    # with open(ff+'.pkl','wb') as F:
    #     pkl.dump({'hullstereovisibility':hullstereovisibility,
    #                      'hullLowErr':hullLowErr,'hulldetection':hulldetection,
    #                      'f':f,'sensID':sens['id'],'b':b,
    #                      'camera_insideFlags':camera_insideFlags,'cost_FOV':cost_FOV,
    #                      'total_max_depth_error':total_max_depth_error,
    #                      'cost_LowErr':cost_LowErr,'pixel_cost_flg_per_camera':pixel_cost_flg_per_camera,
    #                      'cost_detection':cost_detection},F)

#%%

###############  2 Cameras ############################   
pc=ParallelConsumer(getmetrics,Nproc=5,maxInQ=1000000)


Sols2cam=[]
for b in np.arange(1*cm2m,max_baseline,5*cm2m):
    cameras_centers = [np.array([np.round(-b/2,3),0,0]),np.array([np.round(b/2,3),0,0])]
    
    # iterate over cameras
    for sens in sensors:
        for f in flens:
            print(b,sens['id'],f)
            # sol = getmetrics(sens,f,cameras_centers)
            pc.pushInputArg((sens,f,cameras_centers))
        

for sol in pc.iterateOutput():
    f = sol['f']
    cameras_centers = sol['cameras_centers']
    bs = ",".join([str(cc[0]) for cc in cameras_centers])
    sensID = sol['sensID']
    for sens in sensors:
        if sens['id'] == sensID:
            break

    
    print(bs,sensID,f)
    
    sensID = sol['sensID']
    
    ff = "TIDAR/2Cam/2Cam b:(%s), f:%f, sensID:%d"%(bs.replace(".",';'),f,sensID)
    
    hullstereovisibilitypts = sol['hullstereovisibilitypts']
    hullLowErrpts = sol['hullLowErrpts']
    hulldetectionpts = sol['hulldetectionpts']

    # fig=plt.figure()
    # plt.plot(hullstereovisibilitypts[:,0], hullstereovisibilitypts[:,1], 'b^-', lw=2,label='stereo-visible')
    # if len(hullLowErrpts)>0:
    #     plt.plot(hullLowErrpts[:,0], hullLowErrpts[:,1], 'go-', lw=2,label='DepthErr')
    # plt.plot(hulldetectionpts[:,0], hulldetectionpts[:,1], 'rs-', lw=2,label='detection')
    # plt.legend()
    # plt.title("b:%s, f:%f, pxs:%f, npx:%d, npy:%d"%(bs,f,sens['pxs'],sens['w']/sens['pxs'],sens['h']/sens['pxs']))
    # plt.xlim((-FOV_width/2-10,FOV_width/2+10))
    # plt.ylim((0,FOV_depth))
    # plt.grid()
    # plt.show()


    # plt.savefig(ff,format='png')
    # plt.close(fig)

    # with open(ff+'.pkl','wb') as F:
    #     pkl.dump({'hullstereovisibility':hullstereovisibility,
    #                      'hullLowErr':hullLowErr,'hulldetection':hulldetection,
    #                      'f':f,'sensID':sens['id'],'b':b,
    #                      'camera_insideFlags':camera_insideFlags,'cost_FOV':cost_FOV,
    #                      'total_max_depth_error':total_max_depth_error,
    #                      'cost_LowErr':cost_LowErr,'pixel_cost_flg_per_camera':pixel_cost_flg_per_camera,
    #                      'cost_detection':cost_detection},F)

    for kk in ['hullstereovisibility','hullLowErr','hulldetection','camera_insideFlags',
               'total_max_depth_error','pixel_cost_flg_per_camera']:
        sol.pop(kk)

    Sols2cam.append(sol)
    
pc.finish()

with open('2_cams_solutions.pkl','wb') as F:
    pkl.dump([Sols2cam],F)
#%%

###############  3 Cameras ############################   
pc=ParallelConsumer(getmetrics,Nproc=5,maxInQ=1000000)


Sols3cam=[]
for b in np.arange(1*cm2m,max_baseline,5*cm2m):
    cameras_centers = [np.array([np.round(-b/2,3),0,0]),np.array([0,0,0]),np.array([np.round(b/2,3),0,0])]
    
    # iterate over cameras
    for sens in sensors:
        for f in flens:
            print(b,sens['id'],f)
            # sol = getmetrics(sens,f,cameras_centers)
            pc.pushInputArg((sens,f,cameras_centers))
        

for sol in pc.iterateOutput():
    f = sol['f']
    cameras_centers = sol['cameras_centers']
    bs = ",".join([str(cc[0]) for cc in cameras_centers])
    sensID = sol['sensID']
    for sens in sensors:
        if sens['id'] == sensID:
            break

    
    print(bs,sensID,f)
    
    sensID = sol['sensID']
    
    ff = "TIDAR/3Cam/3Cam b:(%s), f:%f, sensID:%d"%(bs.replace(".",';'),f,sensID)
    
    hullstereovisibilitypts = sol['hullstereovisibilitypts']
    hullLowErrpts = sol['hullLowErrpts']
    hulldetectionpts = sol['hulldetectionpts']

    fig=plt.figure()
    plt.plot(hullstereovisibilitypts[:,0], hullstereovisibilitypts[:,1], 'b^-', lw=2,label='stereo-visible')
    if len(hullLowErrpts)>0:
        plt.plot(hullLowErrpts[:,0], hullLowErrpts[:,1], 'go-', lw=2,label='DepthErr')
    plt.plot(hulldetectionpts[:,0], hulldetectionpts[:,1], 'rs-', lw=2,label='detection')
    plt.legend()
    plt.title("b:%s, f:%f, pxs:%f, npx:%d, npy:%d"%(bs,f,sens['pxs'],sens['w']/sens['pxs'],sens['h']/sens['pxs']))
    plt.xlim((-FOV_width/2-10,FOV_width/2+10))
    plt.ylim((0,FOV_depth))
    plt.grid()
    plt.show()


    plt.savefig(ff,format='png')
    plt.close(fig)

    # with open(ff+'.pkl','wb') as F:
    #     pkl.dump({'hullstereovisibility':hullstereovisibility,
    #                      'hullLowErr':hullLowErr,'hulldetection':hulldetection,
    #                      'f':f,'sensID':sens['id'],'b':b,
    #                      'camera_insideFlags':camera_insideFlags,'cost_FOV':cost_FOV,
    #                      'total_max_depth_error':total_max_depth_error,
    #                      'cost_LowErr':cost_LowErr,'pixel_cost_flg_per_camera':pixel_cost_flg_per_camera,
    #                      'cost_detection':cost_detection},F)

    for kk in ['hullstereovisibility','hullLowErr','hulldetection','camera_insideFlags',
               'total_max_depth_error','pixel_cost_flg_per_camera']:
        sol.pop(kk)

    Sols3cam.append(sol)
    
pc.finish()

with open('3_cams_solutions.pkl','wb') as F:
    pkl.dump([Sols3cam],F)            



#%%
###############  4 Cameras ############################                      
pc=ParallelConsumer(getmetrics,Nproc=7,maxInQ=1000000)


Sols4cam=[]
for b1 in np.arange(1*cm2m,max_baseline,10*cm2m):
    for b2 in np.arange(1*cm2m,max_baseline,10*cm2m):
        if b2<=b1:
            continue
        cameras_centers = [np.array([np.round(-b2/2,3),0,0]),np.array([np.round(-b1/2,3),0,0]),np.array([np.round(b1/2,3),0,0]),np.array([np.round(b2/2,3),0,0])]
        
        # iterate over cameras
        for sens in sensors:
            for f in flens:
                print(b1,b2,sens['id'],f)
                # sol = getmetrics(sens,f,cameras_centers)
                pc.pushInputArg((sens,f,cameras_centers))
        

for sol in pc.iterateOutput():
    f = sol['f']
    cameras_centers = sol['cameras_centers']
    bs = ",".join([str(cc[0]) for cc in cameras_centers])
    sensID = sol['sensID']
    for sens in sensors:
        if sens['id'] == sensID:
            break

    
    print(bs,sensID,f)
    
    sensID = sol['sensID']
    
    ff = "TIDAR/4Cam/4Cam b:(%s), f:%f, sensID:%d"%(bs.replace(".",';'),f,sensID)
    
    hullstereovisibilitypts = sol['hullstereovisibilitypts']
    hullLowErrpts = sol['hullLowErrpts']
    hulldetectionpts = sol['hulldetectionpts']

    fig=plt.figure()
    plt.plot(hullstereovisibilitypts[:,0], hullstereovisibilitypts[:,1], 'b^-', lw=2,label='stereo-visible')
    if len(hullLowErrpts)>0:
        plt.plot(hullLowErrpts[:,0], hullLowErrpts[:,1], 'go-', lw=2,label='DepthErr')
    plt.plot(hulldetectionpts[:,0], hulldetectionpts[:,1], 'rs-', lw=2,label='detection')
    plt.legend()
    plt.title("b:%s, f:%f, pxs:%f, npx:%d, npy:%d"%(bs,f,sens['pxs'],sens['w']/sens['pxs'],sens['h']/sens['pxs']))
    plt.xlim((-FOV_width/2-10,FOV_width/2+10))
    plt.ylim((0,FOV_depth))
    plt.grid()
    plt.show()


    plt.savefig(ff,format='png')
    plt.close(fig)

    # with open(ff+'.pkl','wb') as F:
    #     pkl.dump({'hullstereovisibility':hullstereovisibility,
    #                      'hullLowErr':hullLowErr,'hulldetection':hulldetection,
    #                      'f':f,'sensID':sens['id'],'b':b,
    #                      'camera_insideFlags':camera_insideFlags,'cost_FOV':cost_FOV,
    #                      'total_max_depth_error':total_max_depth_error,
    #                      'cost_LowErr':cost_LowErr,'pixel_cost_flg_per_camera':pixel_cost_flg_per_camera,
    #                      'cost_detection':cost_detection},F)

    for kk in ['hullstereovisibility','hullLowErr','hulldetection','camera_insideFlags',
               'total_max_depth_error','pixel_cost_flg_per_camera']:
        sol.pop(kk)

    Sols4cam.append(sol)
    
pc.finish()

with open('4_cams_solutions.pkl','wb') as F:
    pkl.dump([Sols4cam],F)
            
#%%

with open('TIDAR/2_cams_solutions.pkl','rb') as F:
    [Sols2cam] = pkl.load(F)

with open('TIDAR/3_cams_solutions.pkl','rb') as F:
    [Sols3cam] = pkl.load(F)

with open('TIDAR/4_cams_solutions.pkl','rb') as F:
    [Sols4cam] = pkl.load(F)


reqcols = ['#Cams','f', 'sensID',  'cost_FOV', 'cost_LowErr', 'cost_detection', 
           'minStereoErrDepth','maxStereoErrDepth','maxStereoErrtheta',
           'maxDetDepth', 'minDetDepth', 'maxDettheta', 
           'max_baseline', 'image_path'] 


for sett,SolsXcam in [['TIDAR/2Cam/2Cam',Sols2cam],['TIDAR/3Cam/3Cam',Sols3cam],['TIDAR/4Cam/4Cam',Sols4cam]]:
    print(sett)
    for i in range(len(SolsXcam)):
        sol=SolsXcam[i]
        hullDetpts = sol['hulldetectionpts']
        minDetDepth=np.min(hullDetpts[:,1])
        maxDetDepth=np.max(hullDetpts[:,1])
        
        mX=np.max(hullDetpts[:,0])
        mnZ = np.min( hullDetpts[(hullDetpts[:,0]<mX+2) & (hullDetpts[:,0]>mX-2),1] )
        theta = 180*np.arctan2(mX,mnZ)/np.pi
        
        SolsXcam[i]['minDetDepth'] = np.round(minDetDepth,3)
        SolsXcam[i]['maxDetDepth'] = np.round(maxDetDepth,3) 
        SolsXcam[i]['maxDettheta'] = np.round(theta,3)
        
        ##
        hullLowErrpts = sol['hullLowErrpts']
        if len(hullLowErrpts)==0:
            SolsXcam[i]['minStereoErrDepth'] = 0
            SolsXcam[i]['maxStereoErrDepth'] = 0
            SolsXcam[i]['maxStereoErrtheta'] = 0
        else:
            minStereoErrDepth=np.min(hullLowErrpts[:,1])
            maxStereoErrDepth=np.max(hullLowErrpts[:,1])
            
            mX=np.max(hullLowErrpts[:,0])
            mnZ = np.min( hullLowErrpts[(hullLowErrpts[:,0]<mX+2) & (hullLowErrpts[:,0]>mX-2),1] )
            theta = 180*np.arctan2(mX,mnZ)/np.pi
            
            SolsXcam[i]['minStereoErrDepth'] = np.round(minStereoErrDepth,3)
            SolsXcam[i]['maxStereoErrDepth'] = np.round(maxStereoErrDepth,3) 
            SolsXcam[i]['maxStereoErrtheta'] = np.round(theta,3)
            
        ##
        
        cameras_centers=sol['cameras_centers']
        bs = [cc[0] for cc in cameras_centers]
        maxb = np.max(bs) - np.min(bs)
        SolsXcam[i]['max_baseline'] = np.round(maxb,3)
        
        bs = ",".join([str(cc[0]) for cc in cameras_centers])
        f = sol['f']
        sensID = sol['sensID']
        ff = "%s b:(%s), f:%f, sensID:%d"%(sett,bs.replace(".",';'),f,sensID)
        
        SolsXcam[i]['image_path'] = ff
        
        KK=list(SolsXcam[i].keys())
        for cc in KK:
            if cc not in reqcols:
                SolsXcam[i].pop(cc)


df2=pd.DataFrame(Sols2cam)
df2['#Cams']=2

df3=pd.DataFrame(Sols3cam)
df3['#Cams']=3

df4=pd.DataFrame(Sols4cam)
df4['#Cams']=4

df=pd.concat([df2,df3,df4])
df.to_excel('camAnalysis.xlsx')