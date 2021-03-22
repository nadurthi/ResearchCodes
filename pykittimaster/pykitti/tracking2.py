"""Provides 'tracking', which loads and parses tracking benchmark data."""

import datetime as dt
import glob
import os
from collections import namedtuple
import pandas as pd
from matplotlib import patches
import numpy as np

from . import utils


class AA:
    def __init__(self):
        self.a=1



class KittiTracking:
    """Load and parse tracking benchmark data into a usable format."""

    columns = 'id class truncated occluded alpha x1 y1 x2 y2 xd yd zd x y z roty score'.split()
#    classes = 'Car Van Truck Pedestrian Person_sitting Cyclist Tram Misc DontCare'.split()


    def __init__(self, base_path, sequence, **kwargs):
        """Set the path."""
        self.base_path = base_path
        self.sequence = sequence
        self.frames = kwargs.get('frames', None)

        # Default image file extension is 'png'
        self.imtype = kwargs.get('imtype', 'png')

        # Find all the data files
        self._get_file_lists()
        print('files', len(self.cam2_files))
        # Pre-load data that isn't returned as a generator
        self._load_calib()
        self.oxtsdata = None

    @property
    def nframes(self):
        return len(self.cam2_files)

    def __len__(self):
        """Return the number of frames loaded."""
        return len(self.timestamps)

    @property
    def cam2(self):
        """Generator to read image files for cam2 (RGB left)."""
        return utils.yield_images(self.cam2_files, mode='RGB')

    def get_cam2(self, idx):
        """Read image file for cam2 (RGB left) at the specified index."""
        return utils.load_image(self.cam2_files[idx], mode='RGB')

    @property
    def cam3(self):
        """Generator to read image files for cam0 (RGB right)."""
        return utils.yield_images(self.cam3_files, mode='RGB')

    def get_cam3(self, idx):
        """Read image file for cam3 (RGB right) at the specified index."""
        return utils.load_image(self.cam3_files[idx], mode='RGB')

    @property
    def gray(self):
        """Generator to read monochrome stereo pairs from file.
        """
        return zip(self.cam0, self.cam1)

    def get_gray(self, idx):
        """Read monochrome stereo pair at the specified index."""
        return (self.get_cam0(idx), self.get_cam1(idx))

    @property
    def rgb(self):
        """Generator to read RGB stereo pairs from file.
        """
        return zip(self.cam2, self.cam3)

    def get_rgb(self, idx):
        """Read RGB stereo pair at the specified index."""
        return (self.get_cam2(idx), self.get_cam3(idx))

    @property
    def velo(self):
        """Generator to read velodyne [x,y,z,reflectance] scan data from binary files."""
        # Return a generator yielding Velodyne scans.
        # Each scan is a Nx4 array of [x,y,z,reflectance]
        return utils.yield_velo_scans(self.velo_files)

    def get_velo(self, idx):
        """Read velodyne [x,y,z,reflectance] scan at the specified index."""
        return utils.load_velo_scan(self.velo_files[idx])

    def get_oxts(self, idx):
        """read"""
        if self.oxtsdata is None:
            self.oxtsdata= utils.load_oxts_packets_and_poses([self.oxts_files])
        return self.oxtsdata[idx]

    def _get_file_lists(self):
        """Find and list data files for each sensor."""
        self.cam2_files = sorted(glob.glob(
            os.path.join(self.base_path,
                         'image_02',
                         self.sequence,
                         '*.{}'.format(self.imtype))))
        self.cam3_files = sorted(glob.glob(
            os.path.join(self.base_path,
                         'image_03',
                         self.sequence,
                         '*.{}'.format(self.imtype))))
        self.velo_files = sorted(glob.glob(
            os.path.join(self.base_path,
                         'velodyne',
                         self.sequence,
                         '*.bin')))
        self.oxts_files = os.path.join(self.base_path,
                         'oxts',
                         self.sequence+'.txt')

        # Subselect the chosen range of frames, if any
        if self.frames is not None:
            self.cam0_files = utils.subselect_files(
                self.cam0_files, self.frames)
            self.cam1_files = utils.subselect_files(
                self.cam1_files, self.frames)
            self.cam2_files = utils.subselect_files(
                self.cam2_files, self.frames)
            self.cam3_files = utils.subselect_files(
                self.cam3_files, self.frames)
            self.velo_files = utils.subselect_files(
                self.velo_files, self.frames)

    def _load_calib(self):
        """Load and compute intrinsic and extrinsic calibration parameters."""
        # We'll build the calibration parameters as a dictionary, then
        # convert it to a namedtuple to prevent it from being modified later
        data = {}

        # Load the calibration file
        calib_filepath = os.path.join(self.base_path,'calib',self.sequence + '.txt')
#        print(calib_filepath)
        filedata = utils.read_calib_file2(calib_filepath)
#        print(filedata)
        # Create 3x4 projection matrices: 0 to 0, 0 to 1, 0 to 2, 0 to 3
        P_rect_00 = np.reshape(filedata['P0'], (3, 4))
        P_rect_10 = np.reshape(filedata['P1'], (3, 4))
        P_rect_20 = np.reshape(filedata['P2'], (3, 4))
        P_rect_30 = np.reshape(filedata['P3'], (3, 4))

        data['P_rect_00'] = P_rect_00
        data['P_rect_10'] = P_rect_10
        data['P_rect_20'] = P_rect_20
        data['P_rect_30'] = P_rect_30

        data['R_0rect'] = np.reshape(filedata['R_rect'], (3, 3))
        data['Tr_velo_to_cam'] = np.reshape(filedata['Tr_velo_cam'], (3, 4)) # to cam0
        data['Tr_imu_to_velo'] = np.reshape(filedata['Tr_imu_velo'], (3, 4))

        data['Tr_velo_to_cam'] = np.vstack([data['Tr_velo_to_cam'], [0, 0, 0, 1]])
        data['Tr_imu_to_velo'] = np.vstack([data['Tr_imu_to_velo'], [0, 0, 0, 1]])
        data['R_0rect'] = np.hstack([data['R_0rect'], [[0],[0],[0]] ])
        data['R_0rect'] = np.vstack([data['R_0rect'], [0, 0, 0, 1]])

        # Compute the rectified extrinsics from cam0 to camN
        T1 = np.eye(4)
        T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
        T2 = np.eye(4)
        T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
        T3 = np.eye(4)
        T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

        # Compute the velodyne to rectified camera coordinate
        # Tr is Tr_velo_to_cam (cam0)
        # dot is just matmul

        data['T_rectcam0_velo'] = data['R_0rect'].dot(data['Tr_velo_to_cam'])
        data['T_rectcam1_velo'] = T1.dot(data['T_rectcam0_velo'])
        data['T_rectcam2_velo'] = T2.dot(data['T_rectcam0_velo'])
        data['T_rectcam3_velo'] = T3.dot(data['T_rectcam0_velo'])

        # Compute the camera intrinsics
        data['K_cam0'] = P_rect_00[0:3, 0:3]
        data['K_cam1'] = P_rect_10[0:3, 0:3]
        data['K_cam2'] = P_rect_20[0:3, 0:3]
        data['K_cam3'] = P_rect_30[0:3, 0:3]

        # Compute the stereo baselines in meters by projecting the origin of
        # each camera frame into the velodyne frame and computing the distances
        # between them
        p_cam = np.array([0, 0, 0, 1])
        p_velo0 = np.linalg.inv(data['T_rectcam0_velo']).dot(p_cam)
        p_velo1 = np.linalg.inv(data['T_rectcam1_velo']).dot(p_cam)
        p_velo2 = np.linalg.inv(data['T_rectcam2_velo']).dot(p_cam)
        p_velo3 = np.linalg.inv(data['T_rectcam3_velo']).dot(p_cam)

        # base lines in meters
        data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
        data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)   # rgb baseline

        self.calib = namedtuple('CalibData', data.keys())(*data.values())

    def readlabel(self):
        """
        label in left color cam 2
        columns = 'id class truncated occluded alpha x1 y1 x2 y2 xd yd zd x y z roty score'.split()
        classes = 'Car Van Truck Pedestrian Person_sitting Cyclist Tram Misc DontCare'.split()

        """
        columns = ['frame', 'trackId','classtype', 'truncated', 'occluded', 'alpha',
                   'bbx1', 'bby1','bbx2', 'bby2',
                   'h', 'w', 'l',
                   'tx', 'ty', 'tz',
                   'roty', 'score']
        label_filepath = os.path.join(self.base_path,'label_02',self.sequence + '.txt')
        self.dflabel = pd.read_csv(label_filepath,sep=' ',header=None,names= columns,index_col =None)
        self.classlabels = self.dflabel['classtype'].unique().tolist()

        return self.dflabel.copy()

##Values    Name      Description
#----------------------------------------------------------------------------
#   1    frame        Frame within the sequence where the obj appearers
#   1    track id     Unique tracking id of this obj within this sequence
#   1    type         Describes the type of obj: 'Car', 'Van', 'Truck',
#                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
#                     'Misc' or 'DontCare'
#   1    truncated    Integer (0,1,2) indicating the level of truncation.
#                     Note that this is in contrast to the obj detection
#                     benchmark where truncation is a float in [0,1].
#   1    occluded     Integer (0,1,2,3) indicating occlusion state:
#                     0 = fully visible, 1 = partly occluded
#                     2 = largely occluded, 3 = unknown
#   1    alpha        Observation angle of obj, ranging [-pi..pi]
#   4    bbox         2D bounding box of obj in the image (0-based index):
#                     contains left, top, right, bottom pixel coordinates
#   3    dimensions   3D obj dimensions: height, width, length (in meters)
#   3    location     3D obj location x,y,z in camera coordinates (in meters)
#   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
#   1    score        Only for results: Float, indicating confidence in
#                     detection, needed for p/r curves, higher is better.


#The coordinates in the camera coordinate system can be projected in the image
#by using the 3x4 projection matrix in the calib folder, where for the left
#color camera for which the images are provided, P2 must be used. The
#difference between rotation_y and alpha is, that rotation_y is directly
#given in camera coordinates, while alpha also considers the vector from the
#camera center to the object center, to compute the relative orientation of
#the object with respect to the camera. For example, a car which is facing
#along the X-axis of the camera coordinate system corresponds to rotation_y=0,
#no matter where it is located in the X/Z plane (bird's eye view), while
#alpha is zero only, when this object is located along the Z-axis of the
#camera. When moving the car away from the Z-axis, the observation angle
#(\alpha) will change.

def drawBox2D(ax,obj):
    #set styles for occlusion and truncation
    occ_col    = ['g','y','r','w']
    trun_style = ['-','--']

    # draw regular objs
    if obj.classtype != 'DontCare':

      # show rectangular bounding boxes
#      pos = [obj.bbx1,obj.bby1,obj.bbx2-obj.bbx1+1,obj.bby2-obj.bby1+1]
      trc = int(obj.truncated>0.1)
      width = obj.bbx2-obj.bbx1+1
      height = obj.bby2-obj.bby1+1
      rect = patches.Rectangle([obj.bbx1,obj.bby1], width, height, angle=0.0,fill=False,
                        edgecolor = occ_col[obj.occluded],
                linewidth=3,linestyle=trun_style[trc])
      ax.add_patch(rect)


#      rectangle('Position',pos,'EdgeColor',occ_col{obj.occlusion+1},...
#                'LineWidth',3,'LineStyle',trun_style{trc},'parent',h(1).axes)
#      rectangle('Position',pos,'EdgeColor','b', 'parent', h(1).axes)

#      % draw label
      label_text = '%s - %d\n%1.1f rad'%(obj.classtype,obj.trackId,obj.alpha)
      x = (obj.bbx1+obj.bbx2)/2
      y = obj.bby1
      ax.annotate(label_text,(x,max(y-5,40)),color = occ_col[obj.occluded],
                  backgroundcolor = 'k', horizontalalignment = 'center',
                  verticalalignment = 'bottom',fontweight='bold',fontsize = 8)

#      text(x,max(y-5,40),label_text,'color',occ_col{obj.occlusion+1},...
#           'BackgroundColor','k','HorizontalAlignment','center',...
#           'VerticalAlignment','bottom','FontWeight','bold',...
#           'FontSize',8,'parent',h(1).axes)

    # % draw don't care regions
    else:
        trc = int(obj.truncated>0.1)
        width = obj.bbx2-obj.bbx1+1
        height = obj.bby2-obj.bby1+1
        rect = patches.Rectangle([obj.bbx1,obj.bby1], width, height, angle=0.0,fill=False,
                        edgecolor = 'c',
                linewidth=2,linestyle='-')
        ax.add_patch(rect)

def computeBox3D(obj,P):
#    % takes an obj and a projection matrix (P) and projects the 3D
#    % bounding box into the image plane.

#    % index for 3D bounding box faces
    face_idx = np.array([[ 1,2,6,5],#   % front face
                 [2,3,7,6],#   % left face
                 [3,4,8,7],#   % back face
                 [4,1,5,8]]) # % right face
    face_idx=face_idx-1
#    % compute rotational matrix around yaw axis
    R = np.array([[+np.cos(obj.roty), 0, +np.sin(obj.roty)],
                      [ 0, 1,               0],
         [-np.sin(obj.roty), 0, +np.cos(obj.roty)]])

#    % 3D bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h

#    % 3D bounding box corners
    x_corners = np.array( [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2] )
    y_corners = np.array([0,0,0,0,-h,-h,-h,-h] )
    z_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2] )

#    % rotate and translate 3D bounding box
    corners_3D = np.matmul( R,np.vstack([x_corners,y_corners,z_corners]) )
    corners_3D[0,:] = corners_3D[0,:] + obj.tx
    corners_3D[1,:] = corners_3D[1,:] + obj.ty
    corners_3D[2,:] = corners_3D[2,:] + obj.tz

#    % only draw 3D bounding box for objs in front of the camera
    if any(corners_3D[2,:]<0.1):
      corners_2D = []
      return [corners_2D,face_idx]
#    end

#    % project the 3D bounding box into the image plane
    corners_2D = projectToImage(corners_3D, P)

    return [corners_2D,face_idx]

def projectToImage(pts_3D, P):
#    % PROJECTTOIMAGE projects 3D points in given coordinate system in the image
#    % plane using the given projection matrix P.
#    %
#    % Usage: pts_2D = projectToImage(pts_3D, P)
#    %   input: pts_3D: 3xn matrix
#    %          P:      3x4 projection matrix
#    %   output: pts_2D: 2xn matrix
#    %
#    % last edited on: 2012-02-27
#    % Philip Lenz - lenz@kit.edu


#    % project in image
    pts_2D = np.matmul(P , np.vstack([pts_3D, np.ones( pts_3D.shape[1] ) ] ) )
#    % scale projected points
    pts_2D[0,:] = np.divide( pts_2D[0,:], pts_2D[2,:] )
    pts_2D[1,:] = np.divide( pts_2D[1,:],pts_2D[2,:] )
#    pts_2D[2,:] = []
    return pts_2D[0:2,:]


def computeOrientation3D(obj,P):
#    % takes an obj and a projection matrix (P) and projects the 3D
#    % obj orientation vector into the image plane.

#    % compute rotational matrix around yaw axis
    R = np.array([[np.cos(obj.roty),  0, np.sin(obj.roty)],
         [0,               1,              0],
         [-np.sin(obj.roty), 0, np.cos(obj.roty)]])

#    % orientation in obj coordinate system
    orientation_3D = np.array( [[0.0, obj.l],
                      [0.0, 0.0],
                      [0.0, 0.0]] )

#    % rotate and translate in camera coordinate system, project in image
    orientation_3D      = np.matmul(R,orientation_3D)
    orientation_3D[0,:] = orientation_3D[0,:] + obj.tx
    orientation_3D[1,:] = orientation_3D[1,:] + obj.ty
    orientation_3D[2,:] = orientation_3D[2,:] + obj.tz

#    % vector behind image plane?
    if any(orientation_3D[2,:]<0.1):
      orientation_2D = []
      return orientation_2D

#    % project orientation into the image plane
    orientation_2D = projectToImage(orientation_3D,P)

    return orientation_2D

def drawBox3D(ax,obj,corners,face_idx,orientation):
    #% set styles for occlusion and truncation
    occ_col    = ['g','y','r','w']
    trun_style = ['-','--']
    trc        = int(obj.truncated>0.1)

#  % draw projected 3D bounding boxes
    if len(corners)>0:
        for f in range(4):
            ax.plot(np.hstack([corners[0,face_idx[f,:]],corners[0,face_idx[f,0]]])+1,
                     np.hstack([corners[1,face_idx[f,:]],corners[1,face_idx[f,0]]])+1,
                     color=occ_col[obj.occluded],linewidth=3,linestyle=trun_style[trc]
                     )
            # ax.plot(np.hstack([corners[0,face_idx[f,:]]+1,corners[0,face_idx[f,0]]+1]),
            #          np.hstack([corners[1,face_idx[f,:]]+1,corners[1,face_idx[f,0]]+1]),
            #          color='b',linewidth=1
            #          )

#            line([corners(1,face_idx(f,:)),corners(1,face_idx(f,1))]+1,...
#   [corners(2,face_idx(f,:)),corners(2,face_idx(f,1))]+1,...
#   'parent',h(2).axes, 'color',occ_col{obj.occlusion+1},...
#   'LineWidth',3,'LineStyle',trun_style{trc})
#
#    line([corners(1,face_idx(f,:)),corners(1,face_idx(f,1))]+1,...
#   [corners(2,face_idx(f,:)),corners(2,face_idx(f,1))]+1,...
#   'parent',h(2).axes,'color','b','LineWidth',1)


#  % draw orientation vector
    drawOrient=False
    if drawOrient:
        if len(orientation)>0 :
            ax.plot(np.hstack([orientation[0,:]+1,orientation[0,:]+1]),
             np.hstack([orientation[1,:]+1,orientation[1,:]+1]),color='w',lineWidth=4)
    
            ax.plot(np.hstack([orientation[0,:]+1,orientation[0,:]+1]),
             np.hstack([orientation[1,:]+1,orientation[1,:]+1]),color='k',lineWidth=2)




def to_array_list(df, length=None, by_id=True):
    """Converts a dataframe to a list of arrays, with one array for every unique index entry.
    Index is assumed to be 0-based contiguous. If there is a missing index entry, an empty
    numpy array is returned for it.
    Elements in the arrays are sorted by their id.
    :param df:
    :param length:
    :return:
    """

    if by_id:
        assert 'id' in df.columns

        # if `id` is the only column, don't sort it (and don't remove it)
        if len(df.columns) == 1:
            by_id = False

    idx = df.index.unique()
    if length is None:
        length = max(idx) + 1

    l = [np.empty(0) for _ in range(length)]
    for i in idx:
        a = df.loc[i]
        if by_id:
            if isinstance(a, pd.Series):
                a = a[1:]
            else:
                a = a.copy().set_index('id').sort_index()

        l[i] = a.values.reshape((-1, a.shape[-1]))
    return np.asarray(l)


# TODO: Acknowledge this is from HART
class KittiTrackingLabels:
    """Kitt Tracking Label parser. It can limit the maximum number of objs per track,
    filter out objs with class "DontCare", or retain only those objs present
    in a given frame.
    """

    columns = 'id class truncated occluded alpha x1 y1 x2 y2 xd yd zd x y z roty score'.split()
    classes = 'Car Van Truck Pedestrian Person_sitting Cyclist Tram Misc DontCare'.split()

    def __init__(
            self,
            path_or_df,
            bbox_with_size=True,
            remove_dontcare=True,
            split_on_reappear=True):

        if isinstance(path_or_df, pd.DataFrame):
            self._df = path_or_df
        else:
            if not os.path.exists(path_or_df):
                raise ValueError('File {} doesn\'t exist'.format(path_or_df))

            self._df = pd.read_csv(path_or_df, sep=' ', header=None,
                                   index_col=0, skip_blank_lines=True)

            # Detection files have 1 more column than label files
            # label file has 16 columns
            # detection file has 17 columns (the last column is score)
            # Here it checks the number of columns the df has and sets the
            # column names on the df accordingly
            self._df.columns = self.columns[:len(self._df.columns)]

        self.bbox_with_size = bbox_with_size

        if remove_dontcare:
            self._df = self._df[self._df['class'] != 'DontCare']

        for c in self._df.columns:
            self._convert_type(c, np.float32, np.float64)
            self._convert_type(c, np.int32, np.int64)

        # TODO: Add occlusion filtering back in
        truncated_threshold = (0, 2.)
        occluded_threshold = (0, 3.)
        # if not nest.is_sequence(occluded_threshold):
        #     occluded_threshold = (0, occluded_threshold)
        #
        # if not nest.is_sequence(truncated_threshold):
        #     truncated_threshold = (0, truncated_threshold)
        # TODO: Add occlusion filteringinback in
        # self._df = self._df[self._df['occluded'] >= occluded_threshold[0]]
        # self._df = self._df[self._df['occluded'] <= occluded_threshold[1]]
        #
        # self._df = self._df[self._df['truncated'] >= truncated_threshold[0]]
        # self._df = self._df[self._df['truncated'] <= truncated_threshold[1]]

        # make 0-based contiguous ids
        ids = self._df.id.unique()
        offset = max(ids) + 1
        id_map = {
            id: new_id for id,
            new_id in zip(
                ids,
                np.arange(
                    offset,
                    len(ids) +
                    offset))}
        self._df.replace({'id': id_map}, inplace=True)
        self._df.id -= offset

        self.ids = list(self._df.id.unique())
        self.max_objs = len(self.ids)
        self.index = self._df.index.unique()

        if split_on_reappear:
            added_ids = self._split_on_reappear(
                self._df, self.presence, self.ids[-1])
            self.ids.extend(added_ids)
            self.max_objs += len(added_ids)

    def _convert_type(self, column, dest_type, only_from_type=None):
        cond = only_from_type is None or self._df[column].dtype == only_from_type
        if cond:
            self._df[column] = self._df[column].astype(dest_type)

    @property
    def bbox(self):
        bbox = self._df[['id', 'x1', 'y1', 'x2', 'y2']].copy()
        # TODO: Fix this to become x, y, w, h
        if self.bbox_with_size:
            bbox['y2'] -= bbox['y1']
            bbox['x2'] -= bbox['x1']

        """Converts a dataframe to a list of arrays
        :param df:
        :param length:
        :return:
        """

        return to_array_list(bbox)

    @property
    def presence(self):
        return self._presence(self._df, self.index, self.max_objs)

    @property
    def num_objs(self):
        ns = self._df.id.groupby(self._df.index).count()
        absent = list(set(range(len(self))) - set(self.index))
        other = pd.DataFrame([0] * len(absent), absent)
        ns = ns.append(other)
        ns.sort_index(inplace=True)
        return ns.as_matrix().squeeze()

    @property
    def cls(self):
        return to_array_list(self._df[['id', 'class']])

    @property
    def occlusion(self):
        return to_array_list(self._df[['id', 'occluded']])

    @property
    def id(self):
        return to_array_list(self._df['id'])

    def __len__(self):
        return self.index[-1] - self.index[0] + 1

    @classmethod
    def _presence(cls, df, index, n_objs):
        p = np.zeros((index[-1] + 1, n_objs), dtype=bool)
        for i, row in df.iterrows():
            p[i, row.id] = True
        return p

    @classmethod
    def _split_on_reappear(cls, df, p, id_offset):
        """Assign a new identity to an objs that appears after disappearing previously.
        Works on `df` in-place.
        :param df: data frame
        :param p: presence
        :param id_offset: offset added to new ids
        :return:
        """

        next_id = id_offset + 1
        added_ids = []
        nt = p.sum(0)
        start = np.argmax(p, 0)
        end = np.argmax(np.cumsum(p, 0), 0)
        diff = end - start + 1
        is_contiguous = np.equal(nt, diff)
        for id, contiguous in enumerate(is_contiguous):
            if not contiguous:

                to_change = df[df.id == id]
                index = to_change.index
                diff = index[1:] - index[:-1]
                where = np.where(np.greater(diff, 1))[0]
                for w in where:
                    to_change.loc[w + 1:, 'id'] = next_id
                    added_ids.append(next_id)
                    next_id += 1

                df[df.id == id] = to_change

        return added_ids
