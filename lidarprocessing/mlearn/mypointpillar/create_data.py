import copy
import pathlib
import pickle

import numpy as np
from skimage import io as imgio

from lidarprocessing.mlearn.pointcloudcore import box_np_ops
from lidarprocessing.mlearn.pointcloudcore.point_cloud.point_cloud_ops import bound_points_jit
from lidarprocessing.mlearn.mypointpillar import kitti_common as kitti
from lidarprocessing.mlearn.pytorchutils.progress_bar import list_bar as prog_bar
from pykitticustom import tracking2
import random
import os
from skimage import io
import pandas as pd

import json


"""
Note: tqdm has problem in my system(win10), so use my progress bar
try:
    from tqdm import tqdm as prog_bar
except ImportError:
    from second.utils.progress_bar import progress_bar_iter as prog_bar
"""

# with open("lidarprocessing/mlearn/mypointpillar/configs/car.json",'w') as F:
#     json.dump(D,F, indent=2)


    
    
def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


# def _calculate_num_points_in_gt(data_path, infos, relative_path, remove_outside=True, num_features=4):
    
    
#     for info in infos:
#         if relative_path:
#             v_path = str(pathlib.Path(data_path) / info["velodyne_path"])
#         else:
#             v_path = info["velodyne_path"]
#         points_v = np.fromfile(
#             v_path, dtype=np.float32, count=-1).reshape([-1, num_features])
#         rect = info['calib/R0_rect']
#         Trv2c = info['calib/Tr_velo_to_cam']
#         P2 = info['calib/P2']
#         if remove_outside:
#             points_v = box_np_ops.remove_outside_points(points_v, rect, Trv2c, P2,
#                                                         info["img_shape"])

#         # points_v = points_v[points_v[:, 0] > 0]
#         annos = info['annos']
#         num_obj = len([n for n in annos['name'] if n != 'DontCare'])
#         # annos = kitti.filter_kitti_anno(annos, ['DontCare'])
#         dims = annos['dimensions'][:num_obj]
#         loc = annos['location'][:num_obj]
#         rots = annos['rotation_y'][:num_obj]
#         gt_boxes_camera = np.concatenate(
#             [loc, dims, rots[..., np.newaxis]], axis=1)
#         gt_boxes_lidar = box_np_ops.box_camera_to_lidar(
#             gt_boxes_camera, rect, Trv2c)
#         indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
#         num_points_in_gt = indices.sum(0)
#         num_ignored = len(annos['dimensions']) - num_obj
#         num_points_in_gt = np.concatenate(
#             [num_points_in_gt, -np.ones([num_ignored])])
#         annos["num_points_in_gt"] = num_points_in_gt.astype(np.int32)

# ```plain
# └── KITTI_DATASET_ROOT
#        ├── training    <-- 7481 train data
#        |   ├── image_2 <-- for visualization
#        |   ├── calib
#        |   ├── label_2
#        |   ├── velodyne
#        |   └── velodyne_reduced <-- empty directory
#        └── testing     <-- 7580 test data
#            ├── image_2 <-- for visualization
#            ├── calib
#            ├── velodyne
#            └── velodyne_reduced <-- empty directory
# ```

# python create_data.py create_kitti_info_file --data_path=KITTI_DATASET_ROOT


# def create_kitti_info_file(data_path,
#                            save_path=None,
#                            create_trainval=False,
#                            relative_path=True):
def create_kitti_info_file(config):
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    
    
    train_data_path = config['train_input_reader']['kitti_root_path']
    num_features = config['train_input_reader']['num_features']
    dflabels=[]
    for seq in  config['train_input_reader']['seqs']:
        ktrk = tracking2.KittiTracking(train_data_path,seq)
        df=ktrk.readlabel()
        df['dimensions']=0
        # dimensions will convert hwl format to standard lhw(camera) format.
        
        df.reset_index(inplace=True)
        df['seq']=seq
        df['dtype']='train'
        df['img_path']= str(os.path.join(train_data_path,'image_02',seq))+"/"+df['frame'].apply(lambda x: f"{x:06d}.png")
        df['velodyne_path']= str(os.path.join(train_data_path,'velodyne',seq))+"/"+df['frame'].apply(lambda x: f"{x:06d}.bin")
        
        df['calib/P0']=0
        ss=ktrk.calib.P_rect_00
        df['calib/P0']=df['calib/P0'].apply(lambda x:ss)
        
        df['calib/P1']=0
        ss=ktrk.calib.P_rect_10
        df['calib/P1']=df['calib/P1'].apply(lambda x:ss)
        
        df['calib/P2']=0
        ss=ktrk.calib.P_rect_20
        df['calib/P2']=df['calib/P2'].apply(lambda x:ss)
        
        df['calib/P3']=0
        ss=ktrk.calib.P_rect_30
        df['calib/P3']=df['calib/P3'].apply(lambda x:ss)
        
        df['calib/R0_rect']=0
        ss=ktrk.calib.R_0rect
        df['calib/R0_rect']=df['calib/R0_rect'].apply(lambda x:ss)
        
        df['calib/Tr_velo_to_cam']=0
        ss=ktrk.calib.Tr_velo_to_cam
        df['calib/Tr_velo_to_cam']=df['calib/Tr_velo_to_cam'].apply(lambda x:ss)
        
        df['calib/Tr_imu_to_velo']=0
        ss=ktrk.calib.Tr_imu_to_velo
        df['calib/Tr_imu_to_velo']=df['calib/Tr_imu_to_velo'].apply(lambda x:ss)
        
        df['gt_idx']=0
        
        for frame,ds in df.groupby('frame'):
            cc=0
            for ii in ds.index:
                if df.loc[ii,'classtype']!='DontCare':
                    df.loc[ii,'gt_idx']=cc
                    cc+=1
                else:
                    df.loc[ii,'gt_idx']=-1
                    
        rect=ktrk.calib.R_0rect
        Trv2c=ktrk.calib.Tr_velo_to_cam
        P2=ktrk.calib.P_rect_10
        df['img_shape']=0
        df['num_points_in_gt']=0
        for frame,dff in df.groupby('frame'):
            idx=dff.index[0]
            ss= np.array(
                    io.imread(dff.loc[idx,'img_path']).shape[:2], dtype=np.int32)
            df.loc[dff.index,'img_shape']= df.loc[dff.index,'img_shape'].apply(lambda x: ss)
            
            points_v = np.fromfile(
                dff.loc[idx,'velodyne_path'], dtype=np.float32, count=-1).reshape([-1, num_features])
            points_v = box_np_ops.remove_outside_points(points_v, rect, Trv2c, P2,
                                                            df.loc[idx,'img_shape'])
            num_obj = len( dff[dff['classtype']!='DontCare']  )
            
            df.loc[dff.index,'num_obj']=num_obj
            gt_boxes_camera = dff[dff['classtype']!='DontCare'][['tx', 'ty', 'tz','l', 'h', 'w','roty']].values
            # location  tx ty tz is in camera coordinates
            
            # dims = annos['dimensions'][:num_obj]
            # loc = annos['location'][:num_obj]
            # rots = annos['rotation_y'][:num_obj]
            
            # gt_boxes_camera = np.concatenate(
            #     [loc, dims, rots[..., np.newaxis]], axis=1)
            
            # get the location coordinates in lidar coordinates, 
            # after the below operation, the dimensioons and roty do not change
            # gt_boxes_lidar=[tx,ty,tx (location of center in velodyne coords), l,h,w,roty]
            gt_boxes_lidar = box_np_ops.box_camera_to_lidar(
                gt_boxes_camera, rect, Trv2c)
            
            # indicies output is true/false for each point if it is inside any one of the gt 3D boxes in lidar coordinates
            indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
            # indices = #points(rows) x #gtboxes (cols)
            # numbre of points in each gt box
            num_points_in_gt = indices.sum(0)
            
            num_ignored = len(dff[dff['classtype']=='DontCare'])
            num_points_in_gt = np.concatenate(
                [num_points_in_gt, -np.ones([num_ignored])])
            gg= num_points_in_gt.astype(np.int32)
            df.loc[dff.index,"num_points_in_gt"]=df.loc[dff.index,"num_points_in_gt"].apply(lambda x:gg)
        
        df['dimensions']=df[['l','h','w']].apply(lambda row: [row['l'],row['h'],row['w']],axis=1)
        df['location']=df[['tx', 'ty', 'tz']].apply(lambda row: [row['tx'],row['ty'],row['tz']],axis=1)
        dflabels.append(df)

    print("Generate info. this may take several minutes.")
    dflabels=pd.concat(dflabels)
    dflabels.reset_index(inplace=True)
    
    
    dflabels['seq_frame']=dflabels['seq']+'_'+dflabels['frame'].astype(str)
    
    uidxs = list( dflabels['seq_frame'].unique() )

    random.shuffle(uidxs)
    n=len(uidxs)
    n1=int(np.floor(0.75*n))
    
    train_uidxs = uidxs[:n1]
    eval_uidxs = uidxs[n1:]
    
    idx=dflabels[dflabels['seq_frame'].isin(train_uidxs)].index
    dflabels.loc[idx,'dtype']='train'
    
    idx=dflabels[dflabels['seq_frame'].isin(eval_uidxs)].index
    dflabels.loc[idx,'dtype']='eval'
    
    
    
    
    
    dflabels_path= config['train_input_reader']['dflabels_path']
    path = pathlib.Path(dflabels_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dflabels.to_pickle(dflabels_path)
    
    
    # kitti_infos_train = kitti.get_kitti_image_info(
    #     data_path,
    #     training=True,
    #     velodyne=True,
    #     calib=True,
    #     image_ids=train_img_ids,
    #     relative_path=relative_path)
    
    
    # _calculate_num_points_in_gt(data_path, kitti_infos_train, relative_path)
    # filename = save_path / 'kitti_infos_train.pkl'
    # print(f"Kitti info train file is saved to {filename}")
    # with open(filename, 'wb') as f:
    #     pickle.dump(kitti_infos_train, f)
    # kitti_infos_val = kitti.get_kitti_image_info(
    #     data_path,
    #     training=True,
    #     velodyne=True,
    #     calib=True,
    #     image_ids=val_img_ids,
    #     relative_path=relative_path)
    # _calculate_num_points_in_gt(data_path, kitti_infos_val, relative_path)
    # filename = save_path / 'kitti_infos_val.pkl'
    # print(f"Kitti info val file is saved to {filename}")
    # with open(filename, 'wb') as f:
    #     pickle.dump(kitti_infos_val, f)
    """
    if create_trainval:
        kitti_infos_trainval = kitti.get_kitti_image_info(
            data_path,
            training=True,
            velodyne=True,
            calib=True,
            image_ids=trainval_img_ids,
            relative_path=relative_path)
        filename = save_path / 'kitti_infos_trainval.pkl'
        print(f"Kitti info trainval file is saved to {filename}")
        with open(filename, 'wb') as f:
            pickle.dump(kitti_infos_trainval, f)
    """
    # filename = save_path / 'kitti_infos_trainval.pkl'
    # print(f"Kitti info trainval file is saved to {filename}")
    # with open(filename, 'wb') as f:
    #     pickle.dump(kitti_infos_train + kitti_infos_val, f)

    # kitti_infos_test = kitti.get_kitti_image_info(
    #     data_path,
    #     training=False,
    #     label_info=False,
    #     velodyne=True,
    #     calib=True,
    #     image_ids=test_img_ids,
    #     relative_path=relative_path)
    # filename = save_path / 'kitti_infos_test.pkl'
    # print(f"Kitti info test file is saved to {filename}")
    # with open(filename, 'wb') as f:
    #     pickle.dump(kitti_infos_test, f)


def _create_reduced_point_cloud(config,
                                dflabels,
                                flipXcoord=False):
    #project the pointcloud from velodyne into camera frame so we get only from looking pointcloid
    
    # with open(info_path, 'rb') as f:
    #     kitti_infos = pickle.load(f)
    train_data_path = config['train_input_reader']['kitti_root_path']
    data_path = config['train_input_reader']['kitti_root_path']
    reduced_pcl_data_path = config['train_input_reader']['reduced_pcl_data_path']
    reduced_pcl_data_path=pathlib.Path(reduced_pcl_data_path)
    reduced_pcl_data_path.mkdir(parents=True, exist_ok=True)
    
    # dflabelsframes=dflabels.groupby(['seq','frame'])
    
    for (seq,frame),dflabelsframes in prog_bar(dflabels.groupby(['seq','frame'])):
        print (seq,frame)
        idx=dflabelsframes.index[0]
        v_path = dflabelsframes.loc[idx,'velodyne_path']
        points_v = np.fromfile(
            v_path, dtype=np.float32, count=-1).reshape([-1, 4])
        rect = dflabelsframes.loc[idx,'calib/R0_rect']
        P2 = dflabelsframes.loc[idx,'calib/P2']
        Trv2c = dflabelsframes.loc[idx,'calib/Tr_velo_to_cam']
        # first remove z < 0 points
        # keep = points_v[:, -1] > 0
        # points_v = points_v[keep]
        # then remove outside.
        if flipXcoord:
            points_v[:, 0] = -points_v[:, 0]
        points_v = box_np_ops.remove_outside_points(points_v, rect, Trv2c, P2,
                                                    dflabelsframes.loc[idx,"img_shape"])
        
        v_path=pathlib.Path(v_path)
        save_filename = reduced_pcl_data_path/ seq/ v_path.name
        save_filename.parent.mkdir(parents=True, exist_ok=True)
        
        if flipXcoord:
            save_filename = save_filename.parent /  pathlib.Path("flipXcoord_" + save_filename.name)
            dflabels.loc[dflabelsframes.index,'reduced_pcl_xflip']=str(save_filename)
        else:
            dflabels.loc[dflabelsframes.index,'reduced_pcl']=str(save_filename)

        with open(save_filename, 'w') as f:
            points_v.tofile(f)
        

def create_reduced_point_cloud(config,with_flipXcoord=True):
   
    dflabels_path= config['train_input_reader']['dflabels_path']
    dflabels=pd.read_pickle(dflabels_path)
    
    _create_reduced_point_cloud(config, dflabels)
    # _create_reduced_point_cloud(data_path, dflabels, save_path)
    # _create_reduced_point_cloud(data_path, dflabels, save_path)
    if with_flipXcoord:
        _create_reduced_point_cloud(config, dflabels,flipXcoord=True)

    dflabels.to_pickle(dflabels_path)
    
# info_path=None,
# used_classes=None,
# database_save_path=None,
# db_info_save_path=None,
# relative_path=True,
# lidar_only=False,
# bev_only=False,
# coors_range=None
def create_groundtruth_database(config):
    dflabels_path= config['train_input_reader']['dflabels_path']
    dflabels=pd.read_pickle(dflabels_path)
    reduced_pcl_data_path = config['train_input_reader']['reduced_pcl_data_path']
    # dflabelsframes=dflabels.groupby(['seq','frame']).first().reset_index()
    
    
    groundtruth_save_path=config['train_input_reader']['groundtruth_save_path']
    groundtruth_save_path=pathlib.Path(groundtruth_save_path)
    groundtruth_save_path.mkdir(parents=True, exist_ok=True)
    
    # if info_path is None:
    #     info_path = root_path / 'kitti_infos_train.pkl'
    # if database_save_path is None:
    #     database_save_path = root_path / 'gt_database'
    # else:
    #     database_save_path = pathlib.Path(database_save_path)
        
    # if db_info_save_path is None:
    #     db_info_save_path = root_path / "kitti_dbinfos_train.pkl"
    # database_save_path.mkdir(parents=True, exist_ok=True)
    # with open(info_path, 'rb') as f:
    #     kitti_infos = pickle.load(f)
    
    
    # all_db_infos = {}
    # if used_classes is None:
    #     used_classes = list(kitti.get_classes())
    #     used_classes.pop(used_classes.index('DontCare'))
        
    # for name in used_classes:
    #     all_db_infos[name] = []
    num_features=config['train_input_reader']['num_features'] 
    group_counter = 0

    for (seq,frame),dflabelsframes in prog_bar(dflabels.groupby(['seq','frame'])):
        # dflabels_used = dflabelsframes[dflabelsframes['dtype']=='train']
        ix=dflabelsframes.index[0]
        velodyne_path = dflabelsframes.loc[ix,'velodyne_path']
        
        points = np.fromfile(
            velodyne_path, dtype=np.float32, count=-1).reshape([-1, num_features])

        frame = dflabelsframes.loc[ix,'frame']
        rect = dflabelsframes.loc[ix,'calib/R0_rect']
        P2 = dflabelsframes.loc[ix,'calib/P2']
        Trv2c = dflabelsframes.loc[ix,'calib/Tr_velo_to_cam']
        img_shape=dflabelsframes.loc[ix,'img_shape']
        
        points = box_np_ops.remove_outside_points(points, rect, Trv2c, P2,img_shape)


        # annos = info["annos"]
        # names = annos["name"]
        # bboxes = annos["bbox"]
        # difficulty = annos["difficulty"]
        # gt_idxes = annos["index"]
        bboxes=dflabelsframes[dflabelsframes['classtype']!='DontCare'][['bbx1', 'bby1','bbx2', 'bby2']]
        num_obj = len(dflabelsframes[dflabelsframes['classtype']!='DontCare'])
        
        rbbox_cam = dflabelsframes[dflabelsframes['classtype']!='DontCare'][['tx', 'ty', 'tz','l', 'h', 'w','roty']].values
        rbbox_lidar = box_np_ops.box_camera_to_lidar(rbbox_cam, rect, Trv2c)
        # if bev_only: # set z and h to limits
        #     assert coors_range is not None
        #     rbbox_lidar[:, 2] = coors_range[2]
        #     rbbox_lidar[:, 5] = coors_range[5] - coors_range[2]
        
        group_dict = {}
        # group_ids = np.full([bboxes.shape[0]], -1, dtype=np.int64)
        # if "group_ids" in annos:
        #     group_ids = annos["group_ids"]
        # else:
        #     group_ids = np.arange(bboxes.shape[0], dtype=np.int64)
            
        point_indices = box_np_ops.points_in_rbbox(points, rbbox_lidar)
        
        groundfilepath = groundtruth_save_path /seq
        groundfilepath.mkdir(parents=True, exist_ok=True)
        for i,idx in enumerate(dflabelsframes[dflabelsframes['classtype']!='DontCare'].index):
            gt_idx=dflabelsframes.loc[idx,'gt_idx']
            name=dflabelsframes.loc[idx,'classtype']
            filename = f"{frame}_{name}_{gt_idx}.bin"
            filepath =  groundfilepath/ filename
            
            gt_points = points[point_indices[:, i]]
            
            # as rbbox_lidar[i, :3] is the ref point, substract it
            gt_points[:, :3] -= rbbox_lidar[i, :3]
            # now gt_points are with respect to the center bottom face point.
            
            with open(filepath, 'w') as f:
                gt_points.tofile(f)
                
            
            dflabels.loc[idx,'gt_pt_filename']=filepath
            dflabels.loc[idx,'box3d_lidar'] = pickle.dumps(rbbox_lidar[i])
            dflabels.loc[idx,'num_points_in_gt'] =gt_points.shape[0]
            

    dflabels['box3d_lidar']=dflabels['box3d_lidar'].apply(lambda x: x if pd.isnull(x) else pickle.loads(x))
    
    
    dflabels['difficulty']=1
    
    dflabels_path= config['train_input_reader']['dflabels_path']
    dflabels.to_pickle(dflabels_path)

if __name__=="__main__":
    with open("lidarprocessing/mlearn/mypointpillar/configs/car_voxelnet.json") as F:
        config=json.load(F)
    
    
    create_kitti_info_file(config)
    create_reduced_point_cloud(config,with_flipXcoord=False)
    create_reduced_point_cloud(config,with_flipXcoord=True)
    create_groundtruth_database(config)