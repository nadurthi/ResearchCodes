# -*- coding: utf-8 -*-

import pathlib
import pickle
import time
from functools import partial
import pathlib
import pickle
import time
from collections import defaultdict
from torch.utils.data import Dataset
import numpy as np
from skimage import io as imgio
import pandas as pd

import numpy as np

from lidarprocessing.mlearn.pointcloudcore import box_np_ops
from lidarprocessing.mlearn.pointcloudcore import preprocess as prep
import lidarprocessing.mlearn.mypointpillar.kitti_common as kitti
from lidarprocessing.mlearn.pointcloudcore.geometry import points_in_convex_polygon_3d_jit
from lidarprocessing.mlearn.pointcloudcore.point_cloud.bev_ops import points_to_bev




class KittiDataset(Dataset):
    def __init__(self, dflabels, config=None,training=True,target_assigner=None,voxel_generator=None):
        self.config=config
        self.training=training
        self.voxel_generator=voxel_generator
        self.target_assigner=target_assigner
        #self._kitti_infos = kitti.filter_infos_by_used_classes(infos, class_names)
        self.dflabels = dflabels.sample(frac=1).reset_index(drop=True).copy()

        self._num_point_features = config['model']['num_point_features']

        if self.training:
            self.seqFrames=self.dflabels[(self.dflabels['dtype']=='train')]['seq_frame'].unique()
        else:
            self.seqFrames=self.dflabels[self.dflabels['dtype']!='eval']['seq_frame'].unique()
            
        # self._prep_func = partial(prep_func, anchor_cache=anchor_cache)

    def __len__(self):
        return len(self.seqFrames)

    @property
    def kitti_infos(self):
        return self._kitti_infos

    def __getitem__(self, i):
        ss=self.seqFrames[i]
        dfinfo=self.dflabels[self.dflabels['seq_frame']==ss].copy()
        
       
        if self.training:
            example = prep_pointcloud_train(self.dflabels,dfinfo,self.config, self.voxel_generator,self.target_assigner)
        else:
            example = prep_pointcloud_eval(self.dflabels,dfinfo,self.config, self.voxel_generator,self.target_assigner)
            
            
        image_idx = dfinfo.iloc[0]['frame']
        example["image_idx"] = image_idx
        example["image_shape"] = dfinfo.iloc[0]["image_shape"]
        example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
        
        return example


    @property
    def dataset(self):
        return self._dataset
    
    
def prep_pointcloud_eval(dflabels,dfinfo,config, voxel_generator,target_assigner):
    num_point_features = config['model']['num_point_features']
    
    v_path=dfinfo.iloc[0]['reduced_pcl']
        
    points = np.fromfile(
        str(v_path), dtype=np.float32,
        count=-1).reshape([-1, num_point_features])
    

            
    generate_bev = config['model']['use_bev']
    without_reflectivity = config['model']['without_reflectivity']

    out_size_factor = config['model']['rpn']['layer_strides'][0] // config['model']['rpn']['upsample_strides'][0]

    
    max_voxels=voxel_generator._max_voxels

    remove_outside_points=config['train_input_reader']['remove_outside_points']
    shuffle_points=config['train_input_reader']['shuffle_points']
    gt_rotation_noise = config['train_input_reader']['groundtruth_rotation_uniform_noise']
    gt_loc_noise_std = config['train_input_reader']['groundtruth_localization_noise_std']
    global_rotation_noise = config['train_input_reader']['global_rotation_uniform_noise']
    global_scaling_noise = config['train_input_reader']['global_scaling_uniform_noise']
    global_loc_noise_std = config['train_input_reader']['global_loc_noise_std']
    global_random_rot_range=config['train_input_reader']['global_random_rotation_range_per_object']
    remove_environment=config['train_input_reader']['remove_environment']
    remove_points_after_samplet=config['train_input_reader']['remove_points_after_samplet']
    gt_drop_max_keep=config['train_input_reader']['groundtruth_drop_max_keep_points']
    gt_points_drop=config['train_input_reader']['groundtruth_points_drop_percentage']
    anchor_area_threshold=config['train_input_reader']['anchor_area_threshold']
    
    rect = dfinfo.iloc[0]['calib/R0_rect'].astype(np.float32)
    Trv2c = dfinfo.iloc[0]['calib/Tr_velo_to_cam'].astype(np.float32)
    P2 = dfinfo.iloc[0]['calib/P2'].astype(np.float32)
    
    image_idx = dfinfo.iloc[0]['frame']
    image_shape = dfinfo.iloc[0]["image_shape"]
    
    if remove_outside_points:
        points = box_np_ops.remove_outside_points(points, rect, Trv2c, P2,
                                                  image_shape)
        
    
    class_names=config['eval_input_reader']['class_names']
    
    if shuffle_points:
        # shuffle is a little slow.
        np.random.shuffle(points)
    
    voxel_size = voxel_generator.voxel_size
    pc_range = voxel_generator.point_cloud_range
    grid_size = voxel_generator.grid_size
    # [352, 400]

    voxels, coordinates, num_points = voxel_generator.generate(
        points, max_voxels)

    example = {
        'voxels': voxels,
        'num_points': num_points,
        'coordinates': coordinates,
        "num_voxels": np.array([voxels.shape[0]], dtype=np.int64)
    }
    example.update({
        'rect': rect,
        'Trv2c': Trv2c,
        'P2': P2,
    })
    
    
    # if not lidar_input:
    # get the x-y voxel grid size and finally get the output size (down and up strides)
    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]
    
    ret = target_assigner.generate_anchors(feature_map_size)
    anchors = ret["anchors"]
    anchors = anchors.reshape([-1, 7])
    matched_thresholds = ret["matched_thresholds"]
    unmatched_thresholds = ret["unmatched_thresholds"]
    anchors_bv = box_np_ops.rbbox2d_to_near_bbox(
        anchors[:, [0, 1, 3, 4, 6]])
    example["anchors"] = anchors
    # print("debug", anchors.shape, matched_thresholds.shape)
    # anchors_bv = anchors_bv.reshape([-1, 4])
    
    anchors_mask = None
    if anchor_area_threshold >= 0:
        coors = coordinates
        dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask(
            coors, tuple(grid_size[::-1][1:]))
        dense_voxel_map = dense_voxel_map.cumsum(0)
        dense_voxel_map = dense_voxel_map.cumsum(1)
        anchors_area = box_np_ops.fused_get_anchors_area(
            dense_voxel_map, anchors_bv, voxel_size, pc_range, grid_size)
        anchors_mask = anchors_area > anchor_area_threshold
        # example['anchors_mask'] = anchors_mask.astype(np.uint8)
        example['anchors_mask'] = anchors_mask
        
    if generate_bev:
        bev_vxsize = voxel_size.copy()
        bev_vxsize[:2] /= 2
        bev_vxsize[2] *= 2
        bev_map = points_to_bev(points, bev_vxsize, pc_range,
                                without_reflectivity)
        example["bev_map"] = bev_map
        
    return example



def prep_pointcloud_train(dflabels,dfinfo,config, voxel_generator,target_assigner):
    
    """convert point cloud to voxels, create targets if ground truths 
    exists.
    """
    num_point_features = config['model']['num_point_features']
    
    v_path=dfinfo.iloc[0]['reduced_pcl']
        
    points = np.fromfile(
        str(v_path), dtype=np.float32,
        count=-1).reshape([-1, num_point_features])
    

            
    generate_bev = config['model']['use_bev']
    without_reflectivity = config['model']['without_reflectivity']

    out_size_factor = config['model']['rpn']['layer_strides'][0] // config['model']['rpn']['upsample_strides'][0]

    
    max_voxels=voxel_generator._max_voxels

    remove_outside_points=config['train_input_reader']['remove_outside_points']
    shuffle_points=config['train_input_reader']['shuffle_points']
    gt_rotation_noise = config['train_input_reader']['groundtruth_rotation_uniform_noise']
    gt_loc_noise_std = config['train_input_reader']['groundtruth_localization_noise_std']
    global_rotation_noise = config['train_input_reader']['global_rotation_uniform_noise']
    global_scaling_noise = config['train_input_reader']['global_scaling_uniform_noise']
    global_loc_noise_std = config['train_input_reader']['global_loc_noise_std']
    global_random_rot_range=config['train_input_reader']['global_random_rotation_range_per_object']
    remove_environment=config['train_input_reader']['remove_environment']
    remove_points_after_samplet=config['train_input_reader']['remove_points_after_samplet']
    gt_drop_max_keep=config['train_input_reader']['groundtruth_drop_max_keep_points']
    gt_points_drop=config['train_input_reader']['groundtruth_points_drop_percentage']
    anchor_area_threshold=config['train_input_reader']['anchor_area_threshold']
        
    rect = dfinfo.iloc[0]['calib/R0_rect'].astype(np.float32)
    Trv2c = dfinfo.iloc[0]['calib/Tr_velo_to_cam'].astype(np.float32)
    P2 = dfinfo.iloc[0]['calib/P2'].astype(np.float32)
    
    image_idx = dfinfo.iloc[0]['frame']
    image_shape = dfinfo.iloc[0]["image_shape"]
    
    if remove_outside_points:
        points = box_np_ops.remove_outside_points(points, rect, Trv2c, P2,
                                                  image_shape)
        
    class_names=config['eval_input_reader']['class_names']
    
    # if remove_environment:
    #     dfinfo=dfinfo[dfinfo['classtype'].isin(class_names)]
    #     points = prep.remove_points_outside_boxes(points, gt_boxes)
    # else:
    #     remove_outside_points=config['eval_input_reader']['remove_outside_points']
    #     shuffle_points=config['eval_input_reader']['shuffle_points']
    #     gt_rotation_noise=[-np.pi / 3, np.pi / 3]
    #     gt_loc_noise_std=[1.0, 1.0, 1.0]
    #     global_rotation_noise=[-np.pi / 4, np.pi / 4]
    #     global_scaling_noise=[0.95, 1.05]
    #     global_loc_noise_std=(0.2, 0.2, 0.2)
    #     global_random_rot_range=[0.78, 2.35]
    #     remove_environment=config['eval_input_reader']['remove_environment']
    #     remove_points_after_sample=True
    #     gt_drop_max_keep=10
    #     generate_bev=False,
    #     anchor_area_threshold=1
        

    db_sampler_cfg = config['train_input_reader']['database_sampler']   
    global_random_rotation_range_per_object=db_sampler_cfg['global_random_rotation_range_per_object']
    rate = db_sampler_cfg['rate']    
        
    # db_sampler = DataBaseSamplerV2(dflabels, groups, rate, grot_range)

    loc = dfinfo["location"]
    dims = dfinfo["dimensions"]
    rots = dfinfo["roty"]
    gt_names = dfinfo["classtype"]
    # print(gt_names, len(loc))
    # gt_boxes = np.concatenate(
    #     [loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)

    group_ids=dfinfo["gt_ids"]
    


    # gt_boxes = box_np_ops.box_camera_to_lidar(gt_boxes, rect, Trv2c)
    # gt_boxes_bv = box_np_ops.center_to_corner_box2d(
    #         gt_boxes[:, 0:2], gt_boxes[:, 3:5], gt_boxes[:, 6])
    
    
    
    # creating extra samples for data augmentation for this pointcloud
    # max_sample_num is the maximum number of "class" samples (existing+augmented) in the point cloud sample
    
    D=[]
    for class_name, max_sample_num in db_sampler_cfg['sample_groups'].items():
        if max_sample_num < dfinfo[dfinfo['classtype']==class_name].shape[0]:
            # just use all the dfinfo data for this class_name
            continue
        
        dall = pd.concat([dfinfo]+D)
        gt_boxes = np.vstack([ss for ss in dall["box3d_lidar"]])
        
        n = max_sample_num - dfinfo[dfinfo['classtype']==class_name].shape[0]
        dd=dflabels[(dflabels['classtype']==class_name) & (dflabels['dtype']=='train')]
        ridxs=np.random.choice(dd.index,n)
        dr=dd.loc[ridxs].copy()
        dridx=dr.index
        
                
        # get gt boxes in lidar frame
        sampled_gt_boxes = np.vstack([ss for ss in dr.loc["box3d_lidar"]])
        sampled_gt_boxes_bv = box_np_ops.center_to_corner_box2d(
            sampled_gt_boxes[:, 0:2], sampled_gt_boxes[:, 3:5], sampled_gt_boxes[:, 6])
        
        valid_mask = np.zeros([gt_boxes.shape[0]+sampled_gt_boxes.shape[0]], dtype=np.bool_)
        
        # add noise to each box
        boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0).copy()
        prep.noise_per_object_v3_(boxes, None, valid_mask, 0, 0, 
                                  global_random_rotation_range_per_object, num_try=100)
        # now boxes are noise added
        # boxes=[loc, dims, rots]=['tx', 'ty', 'tz','l', 'h', 'w','roty']
        
        sampled_gt_boxes_new = boxes[gt_boxes.shape[0]:]
        sampled_gt_boxes_new_bv = box_np_ops.center_to_corner_box2d(
            sampled_gt_boxes_new[:, 0:2], sampled_gt_boxes_new[:, 3:5], sampled_gt_boxes_new[:, 6])
        
        total_bv = np.concatenate([gt_boxes_bv, sampled_gt_boxes_new_bv], axis=0)
        # coll_mat = collision_test_allbox(total_bv)
        coll_mat = prep.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False
        
        valid_samples = []
        num_gt=gt_boxes.shape[0]
        num_sampled=sampled_gt_boxes.shape[0]
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                dr.loc[dridx[i - num_gt],"box3d_lidar"][:2] = boxes[i, :2] # as x-y are rotated and translated randomly
                dr.loc[dridx[i - num_gt],"box3d_lidar"][-1] = boxes[i, -1] # the rotated angle
                # sampled_gt_boxes_new is before rotation
                # boxes second part is after rotation
                # is the additional rotation
                dr.loc[dridx[i - num_gt],"rot_transform"]=  boxes[i, -1] - sampled_gt_boxes_new[i - num_gt, -1]
                
                
           
        D.append(dr)
    
    if len(D)>0:
        dfinfo_sampled = pd.concat(D)
        dfinfo_sampled.reset_index(inplace=True)
    
    
        # now get the rotated points for the gt
        s_points_list=[]
        for i in dfinfo_sampled.index:
            rot=dfinfo_sampled.loc[i,'rot_transform']
            sv_path=dfinfo_sampled.loc[i,'gt_pt_filename']
            s_points = np.fromfile(sv_path,dtype=np.float32)
            s_points = s_points.reshape([-1, num_point_features])
            
            s_points[:, :3] = box_np_ops.rotation_points_single_angle(
                    s_points[:, :3], rot, axis=2)
            s_points[:, :3] += dfinfo_sampled.loc[i,"box3d_lidar"][:3]
            s_points_list.append(s_points)
            
        sampled_gt_boxes = np.vstack([ss for ss in dfinfo_sampled["box3d_lidar"]])
        num_sampled = dfinfo_sampled.shape[0]
        
        sampled_points=np.concatenate(s_points_list, axis=0)
        
        if remove_points_after_samplet:
            points = prep.remove_points_in_boxes(points, sampled_gt_boxes)
        
        points = np.concatenate([sampled_points, points], axis=0)
        
        dfinfo=pd.concat([dfinfo,dfinfo_sampled])   
        dfinfo.reset_index(inplace=True)
        
        dfinfo['gt_idx']=np.arange(dfinfo['gt_idx'].shape[0])  
                
    # now create samples 
    # returns
    # sampled_dict = {
    #     "gt_names": np.array([s["name"] for s in sampled]),
    #     "difficulty": np.array([s["difficulty"] for s in sampled]),
    #     "gt_boxes": sampled_gt_boxes,
    #     "points": np.concatenate(s_points_list, axis=0),
    #     "gt_masks": np.ones((num_sampled, ), dtype=np.bool_),
    #     "group_ids": np.arange(gt_boxes.shape[0], gt_boxes.shape[0] + len(sampled))
    # }
        


    gt_boxes = np.vstack([ss for ss in dfinfo.loc["box3d_lidar"]])
    gt_boxes_mask=dfinfo['classtype'].isin(db_sampler_cfg['sample_groups'].keys())
    gt_boxes_mask=gt_boxes_mask.values.astype(bool)
    gt_names=dfinfo['classtype'].values
    
    group_ids = dfinfo['gt_idx'].values

    if without_reflectivity:
        used_point_axes = list(range(num_point_features))
        used_point_axes.pop(3)
        points = points[:, used_point_axes]
    
    pc_range = voxel_generator.point_cloud_range
    
    # if bev_only:  # set z and h to limits
    #     gt_boxes[:, 2] = pc_range[2]
    #     gt_boxes[:, 5] = pc_range[5] - pc_range[2]
    
    # Now add noise to all the boxes (true gt and sampled gt)      
    
    prep.noise_per_object_v3_(
        gt_boxes,
        points,
        gt_boxes_mask,
        rotation_perturb=gt_rotation_noise,
        center_noise_std=gt_loc_noise_std,
        global_random_rot_range=global_random_rot_range,
        group_ids=group_ids,
        num_try=100)
    
    # should remove unrelated objects after noise per object
    gt_boxes = gt_boxes[gt_boxes_mask]
    gt_names = gt_names[gt_boxes_mask]
    if group_ids is not None:
        group_ids = group_ids[gt_boxes_mask]
        
    gt_classes = np.array(
        [class_names.index(n) + 1 for n in gt_names], dtype=np.int32)

    gt_boxes, points = prep.random_flip(gt_boxes, points)
    gt_boxes, points = prep.global_rotation(
        gt_boxes, points, rotation=global_rotation_noise)
    gt_boxes, points = prep.global_scaling_v2(gt_boxes, points,
                                              *global_scaling_noise)

    # Global translation
    gt_boxes, points = prep.global_translate(gt_boxes, points, global_loc_noise_std)

    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    mask = prep.filter_gt_box_outside_range(gt_boxes, bv_range)
    gt_boxes = gt_boxes[mask]
    gt_classes = gt_classes[mask]
    if group_ids is not None:
        group_ids = group_ids[mask]

    # limit rad to [-pi, pi]
    gt_boxes[:, 6] = box_np_ops.limit_period(
        gt_boxes[:, 6], offset=0.5, period=2 * np.pi)

    if shuffle_points:
        # shuffle is a little slow.
        np.random.shuffle(points)

    # [0, -40, -3, 70.4, 40, 1]
    voxel_size = voxel_generator.voxel_size
    pc_range = voxel_generator.point_cloud_range
    grid_size = voxel_generator.grid_size
    # [352, 400]

    voxels, coordinates, num_points = voxel_generator.generate(
        points, max_voxels)

    example = {
        'voxels': voxels,
        'num_points': num_points,
        'coordinates': coordinates,
        "num_voxels": np.array([voxels.shape[0]], dtype=np.int64)
    }
    example.update({
        'rect': rect,
        'Trv2c': Trv2c,
        'P2': P2,
    })
    
    
    # if not lidar_input:
    # get the x-y voxel grid size and finally get the output size (down and up strides)
    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]
    
    ret = target_assigner.generate_anchors(feature_map_size)
    anchors = ret["anchors"]
    anchors = anchors.reshape([-1, 7])
    matched_thresholds = ret["matched_thresholds"]
    unmatched_thresholds = ret["unmatched_thresholds"]
    anchors_bv = box_np_ops.rbbox2d_to_near_bbox(
        anchors[:, [0, 1, 3, 4, 6]])
    example["anchors"] = anchors
    # print("debug", anchors.shape, matched_thresholds.shape)
    # anchors_bv = anchors_bv.reshape([-1, 4])
    
    anchors_mask = None
    if anchor_area_threshold >= 0:
        coors = coordinates
        dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask(
            coors, tuple(grid_size[::-1][1:]))
        dense_voxel_map = dense_voxel_map.cumsum(0)
        dense_voxel_map = dense_voxel_map.cumsum(1)
        anchors_area = box_np_ops.fused_get_anchors_area(
            dense_voxel_map, anchors_bv, voxel_size, pc_range, grid_size)
        anchors_mask = anchors_area > anchor_area_threshold
        # example['anchors_mask'] = anchors_mask.astype(np.uint8)
        example['anchors_mask'] = anchors_mask
        
    if generate_bev:
        bev_vxsize = voxel_size.copy()
        bev_vxsize[:2] /= 2
        bev_vxsize[2] *= 2
        bev_map = points_to_bev(points, bev_vxsize, pc_range,
                                without_reflectivity)
        example["bev_map"] = bev_map
        


    targets_dict = target_assigner.assign(
        anchors,
        gt_boxes,
        anchors_mask,
        gt_classes=gt_classes,
        matched_thresholds=matched_thresholds,
        unmatched_thresholds=unmatched_thresholds)
    example.update({
        'labels': targets_dict['labels'],
        'reg_targets': targets_dict['bbox_targets'],
        'reg_weights': targets_dict['bbox_outside_weights'],
        })
    return example