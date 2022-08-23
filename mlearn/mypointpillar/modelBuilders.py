
from lidarprocessing.mlearn.mypointpillar.models.voxelnet import LossNormType, VoxelNet
from lidarprocessing.mlearn.pytorchutils.torchplus.train import learning_schedules
import torch
from lidarprocessing.mlearn.pytorchutils import losses
import torch
from torch.utils.data import Dataset
import numpy as np
from functools import partial
import pickle
import pandas as pd
from lidarprocessing.mlearn.mypointpillar.dataset import KittiDataset
import lidarprocessing.mlearn.pointcloudcore.preprocess as prep
# from lidarprocessing.mlearn.pointcloudcore.preprocess import DataBasePreprocessor
# from lidarprocessing.mlearn.pointcloudcore.sample_ops import DataBaseSamplerV2
from lidarprocessing.mlearn.pointcloudcore.target_assigner import TargetAssigner
from lidarprocessing.mlearn.pointcloudcore import region_similarity
from lidarprocessing.mlearn.pointcloudcore.anchor_generator import (AnchorGeneratorStride, AnchorGeneratorRange)
from lidarprocessing.mlearn.pointcloudcore.box_coders import (BevBoxCoderTorch,
                                              GroundBox3dCoderTorch)



def anchor_generator_builder(anchor_config):
    """Create optimizer based on config.

  Args:
    optimizer_config: A Optimizer proto message.

  Returns:
    An optimizer and a list of variables for summary.

  Raises:
    ValueError: when using an unsupported input data type.
  """
    
    if 'anchor_generator_stride' in anchor_config.keys():
        config = anchor_config['anchor_generator_stride']
        ag = AnchorGeneratorStride(
            sizes=list(config['sizes']),
            anchor_strides=list(config['strides']),
            anchor_offsets=list(config['offsets']),
            rotations=list(config['rotations']),
            match_threshold=config['matched_threshold'],
            unmatch_threshold=config['unmatched_threshold'],
            class_id=config['class_name'])
        return ag
    elif 'anchor_generator_range' in anchor_config.keys():
        config = anchor_config['anchor_generator_range']
        ag = AnchorGeneratorRange(
            sizes=list(config['sizes']),
            anchor_ranges=list(config['anchor_ranges']),
            rotations=list(config['rotations']),
            match_threshold=config['matched_threshold'],
            unmatch_threshold=config['unmatched_threshold'],
            class_id=config['class_name'])
        return ag
    else:
        raise ValueError(" unknown anchor generator type")
        
def similarity_calculator_builder(similarity_config):
    """Create optimizer based on config.

    Args:
        optimizer_config: A Optimizer proto message.

    Returns:
        An optimizer and a list of variables for summary.

    Raises:
        ValueError: when using an unsupported input data type.
    """
    # config['target_assigner']['region_similarity_calculator']

    if 'rotate_iou_similarity' in similarity_config.keys():
        similarity_calc= region_similarity.RotateIouSimilarity()
    elif 'nearest_iou_similarity' in similarity_config.keys():
        similarity_calc= region_similarity.NearestIouSimilarity()
    elif 'distance_similarity' in similarity_config.keys():
        cfg = similarity_config['distance_similarity']
        return region_similarity.DistanceSimilarity(
            distance_norm=cfg['distance_norm'],
            with_rotation=cfg['with_rotation'],
            rotation_alpha=cfg['rotation_alpha'])
    else:
        raise ValueError("unknown similarity type")
        
        
    # similarity_type = similarity_config.WhichOneof('region_similarity')
    # if similarity_type == 'rotate_iou_similarity':
    #     return region_similarity.RotateIouSimilarity()
    # elif similarity_type == 'nearest_iou_similarity':
    #     return region_similarity.NearestIouSimilarity()
    # elif similarity_type == 'distance_similarity':
    #     cfg = similarity_config.distance_similarity
    #     return region_similarity.DistanceSimilarity(
    #         distance_norm=cfg.distance_norm,
    #         with_rotation=cfg.with_rotation,
    #         rotation_alpha=cfg.rotation_alpha)
    # else:
        # raise ValueError("unknown similarity type")
        
        
def box_coder_builder(boxconfig):
    """Create optimizer based on config.

    Args:
        optimizer_config: A Optimizer proto message.

    Returns:
        An optimizer and a list of variables for summary.

    Raises:
        ValueError: when using an unsupported input data type.
    """
    box_coder_type = list(boxconfig.keys())[0]
    if box_coder_type == 'ground_box3d_coder':
        return GroundBox3dCoderTorch(boxconfig['ground_box3d_coder']['linear_dim'], 
                                     boxconfig['ground_box3d_coder']['encode_angle_vector'])
    elif box_coder_type == 'bev_box_coder':
        return BevBoxCoderTorch(boxconfig['ground_box3d_coder']['linear_dim'], 
                                boxconfig['ground_box3d_coder']['encode_angle_vector'], 
                                boxconfig['ground_box3d_coder']['z_fixed'], 
                                boxconfig['ground_box3d_coder']['h_fixed'])
    else:
        raise ValueError("unknown box_coder type")


def target_assigner_builder(target_assigner_config, bv_range, box_coder):
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """

    anchor_cfg = target_assigner_config['anchor_generators']
    anchor_generators = []
    for a_cfg in anchor_cfg:
        anchor_generator = anchor_generator_builder(a_cfg)
        anchor_generators.append(anchor_generator)
        
    similarity_calc = similarity_calculator_builder(
        target_assigner_config['region_similarity_calculator'])
    positive_fraction = target_assigner_config['sample_positive_fraction']
    
    if positive_fraction < 0:
        positive_fraction = None
        
    target_assigner = TargetAssigner(
        box_coder=box_coder,
        anchor_generators=anchor_generators,
        region_similarity_calculator=similarity_calc,
        positive_fraction=positive_fraction,
        sample_size=target_assigner_config['sample_size'])
    return target_assigner


def modelbuilder(config, voxel_generator,
          target_assigner) :
    """build second pytorch instance.
    """

    vfe_num_filters = list(config['model']['voxel_feature_extractor']['num_filters'])
    vfe_with_distance = config['model']['voxel_feature_extractor']['with_distance']
    grid_size = voxel_generator.grid_size
    dense_shape = [1] + grid_size[::-1].tolist() + [vfe_num_filters[-1]]
    num_class = config['model']['num_class']

    num_input_features = config['model']['num_point_features']
    if config['model']['without_reflectivity']:
        num_input_features = 3
    loss_norm_type_dict = {
        'NormByNumExamples':0 ,
        'NormByNumPositives':1 ,
        'NormByNumPosNeg':2,
    }
    loss_norm_type = loss_norm_type_dict[config['model']['loss_norm_type']]

    losses = buildlosses(config)
    
    encode_rad_error_by_sin = config['model']['encode_rad_error_by_sin']
    cls_loss_ftor, loc_loss_ftor, cls_weight, loc_weight, _ = losses
    pos_cls_weight = config['model']['pos_class_weight']
    neg_cls_weight = config['model']['neg_class_weight']
    direction_loss_weight = config['model']['direction_loss_weight']

    net = VoxelNet(
        dense_shape,
        num_class=num_class,
        vfe_class_name=config['model']['voxel_feature_extractor']['module_class_name'],
        vfe_num_filters=vfe_num_filters,
        middle_class_name=config['model']['middle_feature_extractor']['module_class_name'],
        middle_num_filters_d1=list(
            config['model']['middle_feature_extractor']['num_filters_down1']),
        middle_num_filters_d2=list(
            config['model']['middle_feature_extractor']['num_filters_down2']),
        rpn_class_name=config['model']['rpn']['module_class_name'],
        rpn_layer_nums=list(config['model']['rpn']['layer_nums']),
        rpn_layer_strides=list(config['model']['rpn']['layer_strides']),
        rpn_num_filters=list(config['model']['rpn']['num_filters']),
        rpn_upsample_strides=list(config['model']['rpn']['upsample_strides']),
        rpn_num_upsample_filters=list(config['model']['rpn']['num_upsample_filters']),
        use_norm=True,
        use_rotate_nms=config['model']['use_rotate_nms'],
        multiclass_nms=config['model']['use_multi_class_nms'],
        nms_score_threshold=config['model']['nms_score_threshold'],
        nms_pre_max_size=config['model']['nms_pre_max_size'],
        nms_post_max_size=config['model']['nms_post_max_size'],
        nms_iou_threshold=config['model']['nms_iou_threshold'],
        use_sigmoid_score=config['model']['use_sigmoid_score'],
        encode_background_as_zeros=config['model']['encode_background_as_zeros'],
        use_direction_classifier=config['model']['use_direction_classifier'],
        use_bev=config['model']['use_bev'],
        num_input_features=num_input_features,
        num_groups=config['model']['rpn']['num_groups'],
        use_groupnorm=config['model']['rpn']['use_groupnorm'],
        with_distance=vfe_with_distance,
        cls_loss_weight=cls_weight,
        loc_loss_weight=loc_weight,
        pos_cls_weight=pos_cls_weight,
        neg_cls_weight=neg_cls_weight,
        direction_loss_weight=direction_loss_weight,
        loss_norm_type=loss_norm_type,
        encode_rad_error_by_sin=encode_rad_error_by_sin,
        loc_loss_ftor=loc_loss_ftor,
        cls_loss_ftor=cls_loss_ftor,
        target_assigner=target_assigner,
        voxel_size=voxel_generator.voxel_size,
        pc_range=voxel_generator.point_cloud_range
    )
    return net


def buildlosses(config):
    """Build losses based on the config.

      Builds classification, localization losses and optionally a hard example miner
      based on the config.
    
      Args:
        loss_config: A losses_pb2.Loss object.
    
      Returns:
        classification_loss: Classification loss object.
        localization_loss: Localization loss object.
        classification_weight: Classification loss weight.
        localization_weight: Localization loss weight.
        hard_example_miner: Hard example miner object.
    
      Raises:
        ValueError: If hard_example_miner is used with sigmoid_focal_loss.
      """
    
    # First Classification loss
    
    if 'weighted_sigmoid' in config['loss']['classification_loss'].keys():
        classification_loss= losses.WeightedSigmoidClassificationLoss()
    if 'weighted_sigmoid_focal' in config['loss']['classification_loss'].keys():
        cfg = config['loss']['classification_loss']['weighted_sigmoid_focal']
        if cfg['alpha'] > 0:
          alpha = cfg['alpha']
        else:
          alpha = None
        classification_loss= losses.SigmoidFocalClassificationLoss(gamma=cfg['gamma'],alpha=alpha)

    if 'weighted_softmax' in config['loss']['classification_loss'].keys():
        classification_loss= losses.WeightedSoftmaxClassificationLoss(
            logit_scale=config['loss']['classification_loss']['weighted_softmax']['logit_scale'])
    
    if 'weighted_softmax_focal' in  config['loss']['classification_loss'].keys():
        cfg = config['loss']['classification_loss']['weighted_softmax_focal']
        if cfg['alpha'] > 0:
          alpha = cfg['alpha']
        else:
          alpha = None
        classification_loss=  losses.SoftmaxFocalClassificationLoss(gamma=cfg['gamma'],alpha=alpha)
    
    if 'bootstrapped_sigmoid' in config['loss']['classification_loss'].keys():
        cfg = config['loss']['classification_loss']['bootstrapped_sigmoid']
        classification_loss=  losses.BootstrappedSigmoidClassificationLoss(
            alpha=cfg['alpha'], bootstrap_type=('hard' if cfg['hard_bootstrap'] else 'soft'))
    
    # Now localization loss
    if 'weighted_l2' in config['loss']['localization_loss']:
        code_weight = config['loss']['localization_loss']['weighted_l2']['code_weight']
        if len(code_weight) == 0:
          code_weight = None
          
        localization_loss = losses.WeightedL2LocalizationLoss(code_weight)
        
    if 'weighted_smooth_l1' in config['loss']['localization_loss']:
        code_weight = config['loss']['localization_loss']['weighted_smooth_l1']['code_weight']
        if len(code_weight) == 0:
          code_weight = None
        sigma = config['loss']['localization_loss']['weighted_smooth_l1']['sigma']
        localization_loss = losses.WeightedSmoothL1LocalizationLoss(sigma, code_weight)
        
        

    classification_weight = config['loss']['classification_weight']
    localization_weight = config['loss']['localization_weight']

    return (classification_loss, localization_loss,
            classification_weight,
            localization_weight, False)




def optimizer_builder(config, params, name=None, last_step=-1):
    """Create optimizer based on config.
    
      Args:
        optimizer_config: A Optimizer proto message.
    
      Returns:
        An optimizer and a list of variables for summary.
    
      Raises:
        ValueError: when using an unsupported input data type.
      """
    
    
    if 'rms_prop_optimizer' in config['train_config']['optimizer']:
        cfg = config['train_config']['optimizer']['rms_prop_optimizer']
        optimizer = torch.optim.RMSprop(
            params,
            lr=_get_base_lr_by_lr_scheduler(cfg['learning_rate']),
            alpha=cfg['decay'],
            momentum=cfg['momentum_optimizer_value'],
            eps=cfg['epsilon'],
            weight_decay=cfg['weight_decay'])

        lr_scheduler = _create_learning_rate_scheduler(
          cfg['learning_rate'], optimizer, last_step=last_step)
        
    if 'momentum_optimizer' in config['train_config']['optimizer']:
        cfg = config['train_config']['optimizer']['momentum_optimizer']
        optimizer = torch.optim.SGD(params,
            lr=_get_base_lr_by_lr_scheduler(cfg['learning_rate']),
            momentum=cfg['momentum_optimizer_value'],
            weight_decay=cfg['weight_decay'])
        
        lr_scheduler = _create_learning_rate_scheduler(
          cfg['learning_rate'], optimizer, last_step=last_step)
        
    if 'adam_optimizer' in config['train_config']['optimizer']:
        cfg = config['train_config']['optimizer']['adam_optimizer']
        optimizer = torch.optim.Adam(params,
            lr=_get_base_lr_by_lr_scheduler(cfg['learning_rate']),
            weight_decay=cfg['weight_decay'])
        
        lr_scheduler = _create_learning_rate_scheduler(
          cfg['learning_rate'], optimizer, last_step=last_step)
        


    return optimizer,lr_scheduler


def _get_base_lr_by_lr_scheduler(learning_rate_config):

    if 'constant_learning_rate' in learning_rate_config.keys():
        config = learning_rate_config['constant_learning_rate']
        base_lr = config['learning_rate']

    if 'exponential_decay_learning_rate' in learning_rate_config.keys():
        config = learning_rate_config['exponential_decay_learning_rate']
        base_lr = config['initial_learning_rate']

    if 'manual_step_learning_rate' in learning_rate_config.keys():
        config = learning_rate_config['manual_step_learning_rate']
        base_lr = config['initial_learning_rate']
        if not config['schedule']:
            raise ValueError('Empty learning rate schedule.')

    if 'cosine_decay_learning_rate' in learning_rate_config.keys():
        config = learning_rate_config['cosine_decay_learning_rate']
        base_lr = config['learning_rate_base']


    return base_lr




def _create_learning_rate_scheduler(learning_rate_config, optimizer, last_step=-1):
    """Create optimizer learning rate scheduler based on config.
    
    Args:
      learning_rate_config: A LearningRate proto message.
    
    Returns:
      A learning rate.
    
    Raises:
      ValueError: when using an unsupported input data type.
    """

    if 'constant_learning_rate' in learning_rate_config.keys():
        config = learning_rate_config['constant_learning_rate']
        lr_scheduler = learning_schedules.Constant(
          optimizer, last_step=last_step)
    
    if 'exponential_decay_learning_rate' in learning_rate_config.keys():
        config = learning_rate_config['exponential_decay_learning_rate']
        lr_scheduler = learning_schedules.ExponentialDecay(
          optimizer, config['decay_steps'], 
          config['decay_factor'], config['staircase'], last_step=last_step)
    
    if 'manual_step_learning_rate' in learning_rate_config.keys():
        config = learning_rate_config['manual_step_learning_rate']

        learning_rate_step_boundaries = [x['step'] for x in config['schedule']]
        learning_rate_sequence = [config['initial_learning_rate']]
        learning_rate_sequence += [x['learning_rate'] for x in config['schedule']]
        lr_scheduler = learning_schedules.ManualStepping(
          optimizer, learning_rate_step_boundaries, learning_rate_sequence, 
          last_step=last_step)
    
    if 'cosine_decay_learning_rate' in learning_rate_config.keys():
        config = learning_rate_config['cosine_decay_learning_rate']
        lr_scheduler = learning_schedules.CosineDecayWithWarmup(
          optimizer, config['total_steps'], 
          config['warmup_learning_rate'], config['warmup_steps'], 
          last_step=last_step)

    
    return lr_scheduler



################################
# Data Builder
################################




def dataset_builder(config,
          training,
          voxel_generator,
          target_assigner=None):
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """
    dflabels_path= config['train_input_reader']['dflabels_path']
    dflabels=pd.read_pickle(dflabels_path)
    # if training:
    #     dflabels=dflabels[dflabels['dtype']=='train']
    #     dflabels=dflabels[dflabels['classtype']!='DontCare']
    # else:
    #     dflabels=dflabels[dflabels['dtype']!='train']
    
    dbsamplerconfig=config['train_input_reader']['database_sampler']
    for prep_type in dbsamplerconfig['database_prep_steps']:
        if 'filter_by_difficulty' in prep_type.keys()  :
            cfg = prep_type['filter_by_difficulty']
            dflabels=dflabels[~dflabels["difficulty"].isin(cfg['removed_difficulties'])]

        elif prep_type == 'filter_by_min_num_points':
            cfg = prep_type['filter_by_min_num_points']
            D=[]
            for name, min_num in cfg['min_num_point_pairs'].items():
                df = dflabels[(dflabels['classtype']==name) & (dflabels["num_points_in_gt"] >= min_num)]
                D.append(df)
            dflabels=pd.concat(D)
        else:
            raise ValueError("unknown database prep type")


    
    dataset = KittiDataset(dflabels,config=config,training=training,target_assigner=target_assigner)

    return dataset




