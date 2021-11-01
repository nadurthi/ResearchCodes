
from lidarprocessing.mlearn.mypointpillar.models.voxelnet import LossNormType, VoxelNet
from lidarprocessing.mlearn.pytorchutils.torchplus.train import learning_schedules
import torch
from lidarprocessing.mlearn.pytorchutils.torchplus.train import learning_schedules
import torch
from torch.utils.data import Dataset
import numpy as np
from functools import partial
import pickle

from lidarprocessing.mlearn.mypointpillar.dataset import KittiDataset,prep_pointcloud
import lidarprocessing.mlearn.pointcloudcore.preprocess as prep
from lidarprocessing.mlearn.pointcloudcore.preprocess import DataBasePreprocessor
from lidarprocessing.mlearn.pointcloudcore.sample_ops import DataBaseSamplerV2
from lidarprocessing.mlearn.pointcloudcore.target_assigner import TargetAssigner
from lidarprocessing.mlearn.pointcloudcore import region_similarity
from lidarprocessing.mlearn.pointcloudcore.anchor_generator import (AnchorGeneratorStride, AnchorGeneratorRange)


def anchor_generator_builder(anchor_config):
    """Create optimizer based on config.

  Args:
    optimizer_config: A Optimizer proto message.

  Returns:
    An optimizer and a list of variables for summary.

  Raises:
    ValueError: when using an unsupported input data type.
  """
    ag_type = anchor_config.WhichOneof('anchor_generator')

    if ag_type == 'anchor_generator_stride':
        config = anchor_config.anchor_generator_stride
        ag = AnchorGeneratorStride(
            sizes=list(config.sizes),
            anchor_strides=list(config.strides),
            anchor_offsets=list(config.offsets),
            rotations=list(config.rotations),
            match_threshold=config.matched_threshold,
            unmatch_threshold=config.unmatched_threshold,
            class_id=config.class_name)
        return ag
    elif ag_type == 'anchor_generator_range':
        config = anchor_config.anchor_generator_range
        ag = AnchorGeneratorRange(
            sizes=list(config.sizes),
            anchor_ranges=list(config.anchor_ranges),
            rotations=list(config.rotations),
            match_threshold=config.matched_threshold,
            unmatch_threshold=config.unmatched_threshold,
            class_id=config.class_name)
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
    similarity_type = similarity_config.WhichOneof('region_similarity')
    if similarity_type == 'rotate_iou_similarity':
        return region_similarity.RotateIouSimilarity()
    elif similarity_type == 'nearest_iou_similarity':
        return region_similarity.NearestIouSimilarity()
    elif similarity_type == 'distance_similarity':
        cfg = similarity_config.distance_similarity
        return region_similarity.DistanceSimilarity(
            distance_norm=cfg.distance_norm,
            with_rotation=cfg.with_rotation,
            rotation_alpha=cfg.rotation_alpha)
    else:
        raise ValueError("unknown similarity type")
        
def target_assigner_build(target_assigner_config, bv_range, box_coder):
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """

    anchor_cfg = target_assigner_config.anchor_generators
    anchor_generators = []
    for a_cfg in anchor_cfg:
        anchor_generator = anchor_generator_builder.build(a_cfg)
        anchor_generators.append(anchor_generator)
    similarity_calc = similarity_calculator_builder(
        target_assigner_config.region_similarity_calculator)
    positive_fraction = target_assigner_config.sample_positive_fraction
    if positive_fraction < 0:
        positive_fraction = None
    target_assigner = TargetAssigner(
        box_coder=box_coder,
        anchor_generators=anchor_generators,
        region_similarity_calculator=similarity_calc,
        positive_fraction=positive_fraction,
        sample_size=target_assigner_config.sample_size)
    return target_assigner


def modelbuild(config, voxel_generator,
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
        0: LossNormType.NormByNumExamples,
        1: LossNormType.NormByNumPositives,
        2: LossNormType.NormByNumPosNeg,
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
        rpn_class_name=config['model']['rpn.module_class_name'],
        rpn_layer_nums=list(config['model']['rpn.layer_nums']),
        rpn_layer_strides=list(config['model']['rpn.layer_strides']),
        rpn_num_filters=list(config['model']['rpn.num_filters']),
        rpn_upsample_strides=list(config['model']['rpn.upsample_strides']),
        rpn_num_upsample_filters=list(config['model']['rpn.num_upsample_filters']),
        use_norm=True,
        use_rotate_nms=config['model'].use_rotate_nms,
        multiclass_nms=config['model'].use_multi_class_nms,
        nms_score_threshold=config['model'].nms_score_threshold,
        nms_pre_max_size=config['model'].nms_pre_max_size,
        nms_post_max_size=config['model'].nms_post_max_size,
        nms_iou_threshold=config['model'].nms_iou_threshold,
        use_sigmoid_score=config['model'].use_sigmoid_score,
        encode_background_as_zeros=config['model'].encode_background_as_zeros,
        use_direction_classifier=config['model'].use_direction_classifier,
        use_bev=config['model'].use_bev,
        num_input_features=num_input_features,
        num_groups=config['model'].rpn.num_groups,
        use_groupnorm=config['model'].rpn.use_groupnorm,
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
  classification_loss = _build_classification_loss(
      loss_config.classification_loss)
  localization_loss = _build_localization_loss(
      loss_config.localization_loss)
  classification_weight = loss_config.classification_weight
  localization_weight = loss_config.localization_weight
  hard_example_miner = None
  if loss_config.HasField('hard_example_miner'):
    raise ValueError('Pytorch don\'t support HardExampleMiner')
  return (classification_loss, localization_loss,
          classification_weight,
          localization_weight, hard_example_miner)

def build_faster_rcnn_classification_loss(loss_config):
  """Builds a classification loss for Faster RCNN based on the loss config.

  Args:
    loss_config: A losses_pb2.ClassificationLoss object.

  Returns:
    Loss based on the config.

  Raises:
    ValueError: On invalid loss_config.
  """
  if not isinstance(loss_config, losses_pb2.ClassificationLoss):
    raise ValueError('loss_config not of type losses_pb2.ClassificationLoss.')

  loss_type = loss_config.WhichOneof('classification_loss')

  if loss_type == 'weighted_sigmoid':
    return losses.WeightedSigmoidClassificationLoss()
  if loss_type == 'weighted_softmax':
    config = loss_config.weighted_softmax
    return losses.WeightedSoftmaxClassificationLoss(
        logit_scale=config.logit_scale)

  # By default, Faster RCNN second stage classifier uses Softmax loss
  # with anchor-wise outputs.
  config = loss_config.weighted_softmax
  return losses.WeightedSoftmaxClassificationLoss(
      logit_scale=config.logit_scale)


def _build_localization_loss(loss_config):
  """Builds a localization loss based on the loss config.

  Args:
    loss_config: A losses_pb2.LocalizationLoss object.

  Returns:
    Loss based on the config.

  Raises:
    ValueError: On invalid loss_config.
  """
  if not isinstance(loss_config, losses_pb2.LocalizationLoss):
    raise ValueError('loss_config not of type losses_pb2.LocalizationLoss.')

  loss_type = loss_config.WhichOneof('localization_loss')
  
  if loss_type == 'weighted_l2':
    config = loss_config.weighted_l2
    if len(config.code_weight) == 0:
      code_weight = None
    else:
      code_weight = config.code_weight
    return losses.WeightedL2LocalizationLoss(code_weight)

  if loss_type == 'weighted_smooth_l1':
    config = loss_config.weighted_smooth_l1
    if len(config.code_weight) == 0:
      code_weight = None
    else:
      code_weight = config.code_weight
    return losses.WeightedSmoothL1LocalizationLoss(config.sigma, code_weight)

  raise ValueError('Empty loss config.')


def _build_classification_loss(loss_config):
  """Builds a classification loss based on the loss config.

  Args:
    loss_config: A losses_pb2.ClassificationLoss object.

  Returns:
    Loss based on the config.

  Raises:
    ValueError: On invalid loss_config.
  """
  if not isinstance(loss_config, losses_pb2.ClassificationLoss):
    raise ValueError('loss_config not of type losses_pb2.ClassificationLoss.')

  loss_type = loss_config.WhichOneof('classification_loss')

  if loss_type == 'weighted_sigmoid':
    return losses.WeightedSigmoidClassificationLoss()

  if loss_type == 'weighted_sigmoid_focal':
    config = loss_config.weighted_sigmoid_focal
    # alpha = None
    # if config.HasField('alpha'):
    #   alpha = config.alpha
    if config.alpha > 0:
      alpha = config.alpha
    else:
      alpha = None
    return losses.SigmoidFocalClassificationLoss(
        gamma=config.gamma,
        alpha=alpha)
  if loss_type == 'weighted_softmax_focal':
    config = loss_config.weighted_softmax_focal
    # alpha = None
    # if config.HasField('alpha'):
    #   alpha = config.alpha
    if config.alpha > 0:
      alpha = config.alpha
    else:
      alpha = None
    return losses.SoftmaxFocalClassificationLoss(
        gamma=config.gamma,
        alpha=alpha)

  if loss_type == 'weighted_softmax':
    config = loss_config.weighted_softmax
    return losses.WeightedSoftmaxClassificationLoss(
        logit_scale=config.logit_scale)

  if loss_type == 'bootstrapped_sigmoid':
    config = loss_config.bootstrapped_sigmoid
    return losses.BootstrappedSigmoidClassificationLoss(
        alpha=config.alpha,
        bootstrap_type=('hard' if config.hard_bootstrap else 'soft'))

  raise ValueError('Empty loss config.')





def optimizer_builder(optimizer_config, params, name=None):
    """Create optimizer based on config.

  Args:
    optimizer_config: A Optimizer proto message.

  Returns:
    An optimizer and a list of variables for summary.

  Raises:
    ValueError: when using an unsupported input data type.
  """
    optimizer_type = optimizer_config.WhichOneof('optimizer')
    optimizer = None

    if optimizer_type == 'rms_prop_optimizer':
        config = optimizer_config.rms_prop_optimizer
        optimizer = torch.optim.RMSprop(
            params,
            lr=_get_base_lr_by_lr_scheduler(config.learning_rate),
            alpha=config.decay,
            momentum=config.momentum_optimizer_value,
            eps=config.epsilon,
            weight_decay=config.weight_decay)

    if optimizer_type == 'momentum_optimizer':
        config = optimizer_config.momentum_optimizer
        optimizer = torch.optim.SGD(
            params,
            lr=_get_base_lr_by_lr_scheduler(config.learning_rate),
            momentum=config.momentum_optimizer_value,
            weight_decay=config.weight_decay)

    if optimizer_type == 'adam_optimizer':
        config = optimizer_config.adam_optimizer
        optimizer = torch.optim.Adam(
            params,
            lr=_get_base_lr_by_lr_scheduler(config.learning_rate),
            weight_decay=config.weight_decay)

    if optimizer is None:
        raise ValueError('Optimizer %s not supported.' % optimizer_type)

    if optimizer_config.use_moving_average:
        raise ValueError('torch don\'t support moving average')
    if name is None:
        # assign a name to optimizer for checkpoint system
        optimizer.name = optimizer_type
    else:
        optimizer.name = name
    return optimizer


def _get_base_lr_by_lr_scheduler(learning_rate_config):
    base_lr = None
    learning_rate_type = learning_rate_config.WhichOneof('learning_rate')
    if learning_rate_type == 'constant_learning_rate':
        config = learning_rate_config.constant_learning_rate
        base_lr = config.learning_rate

    if learning_rate_type == 'exponential_decay_learning_rate':
        config = learning_rate_config.exponential_decay_learning_rate
        base_lr = config.initial_learning_rate

    if learning_rate_type == 'manual_step_learning_rate':
        config = learning_rate_config.manual_step_learning_rate
        base_lr = config.initial_learning_rate
        if not config.schedule:
            raise ValueError('Empty learning rate schedule.')

    if learning_rate_type == 'cosine_decay_learning_rate':
        config = learning_rate_config.cosine_decay_learning_rate
        base_lr = config.learning_rate_base
    if base_lr is None:
        raise ValueError(
            'Learning_rate %s not supported.' % learning_rate_type)

    return base_lr



def lr_scheduler_builder(optimizer_config, optimizer, last_step=-1):
  """Create lr scheduler based on config. note that
  lr_scheduler must accept a optimizer that has been restored.

  Args:
    optimizer_config: A Optimizer proto message.

  Returns:
    An optimizer and a list of variables for summary.

  Raises:
    ValueError: when using an unsupported input data type.
  """
  optimizer_type = optimizer_config.WhichOneof('optimizer')

  if optimizer_type == 'rms_prop_optimizer':
    config = optimizer_config.rms_prop_optimizer
    lr_scheduler = _create_learning_rate_scheduler(
      config.learning_rate, optimizer, last_step=last_step)

  if optimizer_type == 'momentum_optimizer':
    config = optimizer_config.momentum_optimizer
    lr_scheduler = _create_learning_rate_scheduler(
      config.learning_rate, optimizer, last_step=last_step)

  if optimizer_type == 'adam_optimizer':
    config = optimizer_config.adam_optimizer
    lr_scheduler = _create_learning_rate_scheduler(
      config.learning_rate, optimizer, last_step=last_step)

  return lr_scheduler

def _create_learning_rate_scheduler(learning_rate_config, optimizer, last_step=-1):
  """Create optimizer learning rate scheduler based on config.

  Args:
    learning_rate_config: A LearningRate proto message.

  Returns:
    A learning rate.

  Raises:
    ValueError: when using an unsupported input data type.
  """
  lr_scheduler = None
  learning_rate_type = learning_rate_config.WhichOneof('learning_rate')
  if learning_rate_type == 'constant_learning_rate':
    config = learning_rate_config.constant_learning_rate
    lr_scheduler = learning_schedules.Constant(
      optimizer, last_step=last_step)

  if learning_rate_type == 'exponential_decay_learning_rate':
    config = learning_rate_config.exponential_decay_learning_rate
    lr_scheduler = learning_schedules.ExponentialDecay(
      optimizer, config.decay_steps, 
      config.decay_factor, config.staircase, last_step=last_step)

  if learning_rate_type == 'manual_step_learning_rate':
    config = learning_rate_config.manual_step_learning_rate
    if not config.schedule:
      raise ValueError('Empty learning rate schedule.')
    learning_rate_step_boundaries = [x.step for x in config.schedule]
    learning_rate_sequence = [config.initial_learning_rate]
    learning_rate_sequence += [x.learning_rate for x in config.schedule]
    lr_scheduler = learning_schedules.ManualStepping(
      optimizer, learning_rate_step_boundaries, learning_rate_sequence, 
      last_step=last_step)

  if learning_rate_type == 'cosine_decay_learning_rate':
    config = learning_rate_config.cosine_decay_learning_rate
    lr_scheduler = learning_schedules.CosineDecayWithWarmup(
      optimizer, config.total_steps, 
      config.warmup_learning_rate, config.warmup_steps, 
      last_step=last_step)

  if lr_scheduler is None:
    raise ValueError('Learning_rate %s not supported.' % learning_rate_type)

  return lr_scheduler



################################
# Data Builder
################################
def build_db_preprocess(db_prep_config):
    prep_type = db_prep_config.WhichOneof('database_preprocessing_step')

    if prep_type == 'filter_by_difficulty':
        cfg = db_prep_config['filter_by_difficulty']
        return prep.DBFilterByDifficulty(list(cfg['removed_difficulties']))
    elif prep_type == 'filter_by_min_num_points':
        cfg = db_prep_config['filter_by_min_num_points']
        return prep.DBFilterByMinNumPoint(dict(cfg['min_num_point_pairs']))
    else:
        raise ValueError("unknown database prep type")


def dbsampler_builder(sampler_config):
    cfg = sampler_config
    groups = list(cfg['sample_groups'])
    prepors = [build_db_preprocess(c) for c in cfg['database_prep_steps']]
    db_prepor = DataBasePreprocessor(prepors)
    rate = cfg['rate']
    grot_range = cfg['global_random_rotation_range_per_object']
    groups = [dict(g['name_to_max_num']) for g in groups]
    info_path = cfg['database_info_path']
    with open(info_path, 'rb') as f:
        db_infos = pickle.load(f)
    grot_range = list(grot_range)
    if len(grot_range) == 0:
        grot_range = None
    sampler = DataBaseSamplerV2(db_infos, groups, db_prepor, rate, grot_range)
    return sampler



def dataset_builder(input_reader_config,
          model_config,
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

    generate_bev = model_config['use_bev']
    without_reflectivity = model_config['without_reflectivity']
    num_point_features = model_config['num_point_features']
    out_size_factor = model_config['rpn']['layer_strides'][0] // model_config['rpn']['upsample_strides'][0]

    cfg = input_reader_config
    db_sampler_cfg = input_reader_config['database_sampler']
    db_sampler = None
    if len(db_sampler_cfg['sample_groups']) > 0:  # enable sample
        db_sampler = dbsampler_builder(db_sampler_cfg)
    u_db_sampler_cfg = input_reader_config['unlabeled_database_sampler']
    u_db_sampler = None
    if len(u_db_sampler_cfg['sample_groups']) > 0:  # enable sample
        u_db_sampler = dbsampler_builder(u_db_sampler_cfg)
    grid_size = voxel_generator.grid_size
    # [352, 400]
    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]

    prep_func = partial(
        prep_pointcloud,
        root_path=cfg.kitti_root_path,
        class_names=list(cfg.class_names),
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        training=training,
        max_voxels=cfg.max_number_of_voxels,
        remove_outside_points=False,
        remove_unknown=cfg.remove_unknown_examples,
        create_targets=training,
        shuffle_points=cfg.shuffle_points,
        gt_rotation_noise=list(cfg.groundtruth_rotation_uniform_noise),
        gt_loc_noise_std=list(cfg.groundtruth_localization_noise_std),
        global_rotation_noise=list(cfg.global_rotation_uniform_noise),
        global_scaling_noise=list(cfg.global_scaling_uniform_noise),
        global_loc_noise_std=(0.2, 0.2, 0.2),
        global_random_rot_range=list(
            cfg.global_random_rotation_range_per_object),
        db_sampler=db_sampler,
        unlabeled_db_sampler=u_db_sampler,
        generate_bev=generate_bev,
        without_reflectivity=without_reflectivity,
        num_point_features=num_point_features,
        anchor_area_threshold=cfg.anchor_area_threshold,
        gt_points_drop=cfg.groundtruth_points_drop_percentage,
        gt_drop_max_keep=cfg.groundtruth_drop_max_keep_points,
        remove_points_after_sample=cfg.remove_points_after_sample,
        remove_environment=cfg.remove_environment,
        use_group_id=cfg.use_group_id,
        out_size_factor=out_size_factor)
    dataset = KittiDataset(
        info_path=cfg.kitti_info_path,
        root_path=cfg.kitti_root_path,
        num_point_features=num_point_features,
        target_assigner=target_assigner,
        feature_map_size=feature_map_size,
        prep_func=prep_func)

    return dataset


class DatasetWrapper(Dataset):
    """ convert our dataset to Dataset class in pytorch.
    """

    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]

    @property
    def dataset(self):
        return self._dataset




