import argparse
import os
import glob
import numpy as np
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
# from waymo_open_dataset import label_pb2 as open_label
import imageio
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
import pickle
from collections import defaultdict
from copy import deepcopy
# import waymo
import cv2
import open3d as o3d


SAVE_INTRINSIC = True
SINGLE_TRACK_INFO_FILE = True
DEBUG = False # If True, processing only the first tfrecord file, and saving with a "_debug" suffix.
MULTIPLE_DIRS = False
# DATADIRS = '/media/ybahat/data/Datasets/Waymo/val/0001'

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datadir')
parser.add_argument('-nd', '--no_data', default=False)
args,_ = parser.parse_known_args()
datadirs = args.datadir
export_data = not args.no_data
# datadirs = DATADIRS
saving_dir = '/'.join(datadirs.split('/')[:-1])
if '.tfrecord' not in datadirs:
    saving_dir = 1*datadirs
    datadirs = glob.glob(datadirs+'/*.tfrecord',recursive=True)
    datadirs = sorted([f for f in datadirs if '.tfrecord' in f])
    MULTIPLE_DIRS = True

if not isinstance(datadirs,list):   datadirs = [datadirs]
if not os.path.isdir(saving_dir):   os.mkdir(saving_dir)

def extract_label_fields(l,dims):
    assert dims in [2,3]
    label_dict = {'c_x':l.box.center_x,'c_y':l.box.center_y,'width':l.box.width,'length':l.box.length,'type':l.type}
    if dims==3:
        label_dict['c_z'] = l.box.center_z
        label_dict['height'] = l.box.height
        label_dict['heading'] = l.box.heading
    return label_dict

def read_intrinsic(intrinsic_params_vector):
    return dict(zip(['f_u', 'f_v', 'c_u', 'c_v', 'k_1', 'k_2', 'p_1', 'p_2', 'k_3'], intrinsic_params_vector))

isotropic_focal = lambda intrinsic_dict: intrinsic_dict['f_u']==intrinsic_dict['f_v']

# datadirs = [os.path.join(args.datadir,
#             'segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord')]
for file_num,file in enumerate(datadirs):
    if SINGLE_TRACK_INFO_FILE:
        tracking_info = {}
    if file_num > 0 and DEBUG:   break
    file_name = file.split('/')[-1].split('.')[0]
    print('Processing file ',file_name)
    if not os.path.isdir(os.path.join(saving_dir, file_name)):   os.mkdir(os.path.join(saving_dir, file_name))
    if not os.path.isdir(os.path.join(saving_dir,file_name, 'images')):   os.mkdir(os.path.join(saving_dir,file_name, 'images'))
    if not os.path.isdir(os.path.join(saving_dir, file_name, 'point_cloud')):   os.mkdir(os.path.join(saving_dir, file_name, 'point_cloud'))
    if not SINGLE_TRACK_INFO_FILE:
        if not os.path.isdir(os.path.join(saving_dir,file_name, 'tracking')):   os.mkdir(os.path.join(saving_dir,file_name, 'tracking'))
    dataset = tf.data.TFRecordDataset(file, compression_type='')
    for f_num, data in enumerate(tqdm(dataset)):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        pose = np.zeros([len(frame.images), 4, 4])
        im_paths = {}
        pcd_paths = {}
        if SAVE_INTRINSIC:
            intrinsic = np.zeros([len(frame.images),9])
        extrinsic = np.zeros_like(pose)
        width,height,camera_labels = np.zeros([len(frame.images)]),np.zeros([len(frame.images)]),defaultdict(dict)
        for im in frame.images:
            saving_name = os.path.join(saving_dir,file_name, 'images','%03d_%s.png'%(f_num,open_dataset.CameraName.Name.Name(im.name)))
            if not DEBUG and export_data:
                im_array = tf.image.decode_jpeg(im.image).numpy()
                # No compression imageio
                # imageio.imwrite(saving_name, im_array, compress_level=0)
                # Less compression imageio
                imageio.imwrite(saving_name, im_array, compress_level=3)
                # Original:
                # imageio.imwrite(saving_name, im_array,)
                # OpenCV Alternative (needs debugging for right colors):
                # cv2.imwrite(saving_name, im_array)
            pose[im.name-1, :, :] = np.reshape(im.pose.transform, [4, 4])
            im_paths[im.name] = saving_name
            extrinsic[im.name-1, :, :] = np.reshape(frame.context.camera_calibrations[im.name-1].extrinsic.transform, [4, 4])
            if SAVE_INTRINSIC:
                intrinsic[im.name-1, :] = frame.context.camera_calibrations[im.name-1].intrinsic
                assert isotropic_focal(read_intrinsic(intrinsic[im.name-1, :])),'Unexpected difference between f_u and f_v.'
            width[im.name-1] = frame.context.camera_calibrations[im.name-1].width
            height[im.name-1] = frame.context.camera_calibrations[im.name-1].height
            for obj_label in frame.projected_lidar_labels[im.name-1].labels:
                camera_labels[im.name][obj_label.id.replace('_'+open_dataset.CameraName.Name.Name(im.name),'')] = extract_label_fields(obj_label,2)
        # Extract point cloud data from stored range images
        laser_calib = np.zeros([len(frame.lasers), 4,4])
        if export_data:
            (range_images, camera_projections, range_image_top_pose) = \
                frame_utils.parse_range_image_and_camera_projection(frame)
            points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame,
                                                                               range_images,
                                                                               camera_projections,
                                                                               range_image_top_pose)
        else:
            points =np.empty([len(frame.lasers), 1])

        laser_mapping = {}
        for (laser, pts) in zip(frame.lasers, points):
            saving_name = os.path.join(saving_dir, file_name, 'point_cloud', '%03d_%s.ply' % (f_num, open_dataset.LaserName.Name.Name(laser.name)))
            if export_data:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                o3d.io.write_point_cloud(saving_name, pcd)
            calib_id = int(np.where(np.array([cali.name for cali in frame.context.laser_calibrations[:5]]) == laser.name)[0])
            laser_calib[laser.name-1, :, :] = np.reshape(frame.context.laser_calibrations[calib_id].extrinsic.transform, [4, 4])
            pcd_paths[laser.name] = saving_name
            laser_mapping.update({open_dataset.LaserName.Name.Name(laser.name): calib_id})

        if 'intrinsic' in tracking_info:
            assert np.all(tracking_info['intrinsic']==intrinsic) and np.all(tracking_info['width']==width) and np.all(tracking_info['height']==height)
        else:
            tracking_info['intrinsic'],tracking_info['width'],tracking_info['height'] = intrinsic,width,height
        dict_2_save = {'per_cam_veh_pose':pose,'cam2veh':extrinsic,'im_paths':im_paths,'width':width,'height':height,
                       'veh2laser':laser_calib, 'pcd_paths': pcd_paths}
        if SAVE_INTRINSIC and not SINGLE_TRACK_INFO_FILE:
            dict_2_save['intrinsic'] = intrinsic
        lidar_labels = {}
        for obj_label in frame.laser_labels:
            lidar_labels[obj_label.id] = extract_label_fields(obj_label,3)
        dict_2_save['lidar_labels'] = lidar_labels
        dict_2_save['camera_labels'] = camera_labels
        dict_2_save['veh_pose'] = np.reshape(frame.pose.transform,[4,4])
        # dict_2_save['lidar2veh'] = np.reshape(frame.context.laser_calibrations['extrinsic'].transform,[4,4])
        dict_2_save['timestamp'] = frame.timestamp_micros
        if SINGLE_TRACK_INFO_FILE:
            tracking_info[(file_num,f_num)] = deepcopy(dict_2_save)
        else:
            with open(os.path.join(saving_dir,file_name, 'tracking','%03d.pkl'%(f_num)),'wb') as f:
                pickle.dump(dict_2_save,f)
    if SINGLE_TRACK_INFO_FILE:
        with open(os.path.join(saving_dir, file_name, 'tracking_info%s.pkl'%('_debug' if DEBUG else '')), 'wb') as f:
            pickle.dump(tracking_info, f)