import os
import time
import imageio
import numpy as np
import random
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf

# TODO: Make waymo data import and work with neural scenes graphs
# from waymo_open_dataset.utils import frame_utils
# from waymo_open_dataset import dataset_pb2 as open_dataset
# from waymo_open_dataset.utils.transform_utils import *
camera_names = {1: 'FRONT',
                2: 'FRONT LEFT',
                3: 'FRONT RIGHT',
                4: 'SIDE LEFT',
                5: 'SIDE RIGHT',}

cameras = [1,2,3]  # [1, 2, 3]


def get_scene_objects(dataset, speed_thresh):
    waymo_obj_meta = {}
    max_n_obj = 0

    for i, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        types = [frame.laser_labels[0].TYPE_PEDESTRIAN,
                 frame.laser_labels[0].TYPE_VEHICLE,
                 frame.laser_labels[0].TYPE_CYCLIST]

        n_obj_frame = 0
        for laser_label in frame.laser_labels:
            if laser_label.type in types:
                id = laser_label.id
                # length, height, width
                dim = np.array([laser_label.box.length, laser_label.box.height, laser_label.box.width])
                speed = np.sqrt(np.sum([laser_label.metadata.speed_x ** 2, laser_label.metadata.speed_y ** 2]))
                acc_x = laser_label.metadata.accel_x

                if speed > speed_thresh:
                    n_obj_frame += 1
                    if id not in waymo_obj_meta:
                        internal_track_id = len(waymo_obj_meta) + 1
                        meta_obj = [id, internal_track_id, laser_label.type, dim]
                        waymo_obj_meta[id] = meta_obj
                    else:
                        if np.sum(waymo_obj_meta[id][3] - dim) > 1e-10:
                            print('Dimension mismatch for same object!')
                            print(id)
                            print(np.sum(waymo_obj_meta[id][2] - dim))

            if n_obj_frame > max_n_obj:
                max_n_obj = n_obj_frame

    return waymo_obj_meta, max_n_obj


def get_frame_objects(laser_labels, v2w_frame_i, waymo_obj_meta, speed_thresh):
    # Get all objects from a frame
    frame_obj_dict = {}

    # TODO: Add Cyclists and pedestrians (like for metadata)
    types = [laser_labels[0].TYPE_VEHICLE]

    for label in laser_labels:
        if label.type in types:
            id = label.id

            if id in waymo_obj_meta:
                # TODO: CHECK vkitti/nerf x, y, z
                waymo2vkitti_vehicle = np.array([[1., 0., 0., 0.],
                                                 [0., 0., -1., 0.],
                                                 [0., 1., 0., 0.],
                                                 [0., 0., 0., 1.]])

                x_v = label.box.center_x
                y_v = label.box.center_y
                z_v = label.box.center_z
                yaw_obj = np.array(label.box.heading)

                R_obj2v = get_yaw_rotation(yaw_obj)
                t_obj_v = tf.constant([x_v, y_v, z_v])
                transform_obj2v = get_transform(tf.cast(R_obj2v, tf.double), tf.cast(t_obj_v, tf.double))
                transform_obj2w = np.matmul(v2w_frame_i, transform_obj2v)
                R = transform_obj2w[:3, :3]

                yaw_aprox = np.arctan2(-R[2, 0], R[0, 0]) # np.arctan2(R[2, 0], R[0, 0]) - np.arctan2(0, 1)
                if yaw_aprox > np.pi:
                    yaw_aprox -= 2*np.pi
                elif yaw_aprox > np.pi:
                    yaw_aprox += 2*np.pi

                yaw_aprox_o = np.arccos(transform_obj2w[0, 0])
                if np.absolute(np.rad2deg(yaw_aprox - yaw_aprox_o)) > 1e-2:
                    a = 0

                # yaw_aprox = yaw_aprox_o

                speed = np.sqrt(np.sum([label.metadata.speed_x ** 2, label.metadata.speed_y ** 2]))
                is_moving = 1. if speed > speed_thresh else 0.

                obj_prop = np.array(
                    [transform_obj2w[0, 3], transform_obj2w[1, 3], transform_obj2w[2, 3], yaw_aprox, 0, 0, is_moving])
                frame_obj_dict[id] = obj_prop

    return frame_obj_dict


def get_camera_pose(v2w_frame_i, calibration):
    # FROM Waymo OD documentation:
    # "Each sensor comes with an extrinsic transform that defines the transform from the
    # sensor frame to the vehicle frame.
    #
    # The camera frame is placed in the center of the camera lens.
    # The x-axis points down the lens barrel out of the lens.
    # The z-axis points up. The y/z plane is parallel to the camera plane.
    # The coordinate system is right handed."

    # Match opengl z --> -x, x --> y, y --> z
    opengl2camera = np.array([[0., 0., -1., 0.],
                              [-1., 0., 0., 0.],
                              [0., 1., 0., 0.],
                              [0., 0., 0., 1.]])
    extrinsic_transform_c2v = np.reshape(calibration.extrinsic.transform, [4, 4])
    extrinsic_transform_c2v = np.matmul(extrinsic_transform_c2v, opengl2camera)
    c2w_frame_i_cam_c = np.matmul(v2w_frame_i, extrinsic_transform_c2v)


    return c2w_frame_i_cam_c

def get_bbox_2d(label_2d):
    center = [label_2d.box.center_x, label_2d.box.center_y]
    dim_box = [label_2d.box.length, label_2d.box.width]

    left = np.ceil(center[0] - dim_box[0] * 0.5)
    right = np.ceil(center[0] + dim_box[0] * 0.5)
    top = np.ceil(center[1] + dim_box[1] * 0.5)
    bottom = np.ceil(center[1] - dim_box[1] * 0.5)

    return np.array([left, right, bottom, top])[None, :]


def load_waymo_od_data(basedir, selected_frames, max_frames=5, use_obj=True, row_id=False):
    """
    :param basedir: Path to segment tfrecord
    :param max_frames:
    :param use_obj:
    :param row_id:
    :return:
    """
    if selected_frames == -1:
        start_frame = 0
        end_frame = 0
    else:
        start_frame = selected_frames[0]
        end_frame = selected_frames[1]

    print('Scene Representation from cameras:')
    for cam in cameras:
        print(camera_names[cam], ',')

    speed_thresh = 5.1


    dataset = tf.data.TFRecordDataset(basedir, compression_type='')

    frames = []

    print('Extracting all moving objects!')
    # Extract all moving objects
    # waymo_obj_meta: object_id, object_type, object_label, color, lenght, height, width
    # max_n_obj: maximum number of objects in a single frame
    waymo_obj_meta, max_n_obj = get_scene_objects(dataset, speed_thresh)

    # All images from cameras specified at the beginning
    images = []

    # Pose of each images camera
    poses = []

    # 2D bounding boxes
    bboxes = []

    # Frame Number, Camera Name, object_id, xyz, angle, ismoving
    visible_objects = []
    max_frame_obj = 0

    count = []
    # Extract all frames from a single tf_record
    for i, data in enumerate(dataset):
        if start_frame <= i <= end_frame:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            
            # FROM Waymo OD documentation:
            # Global Frame/ World Frame
            # The origin of this frame is set to the vehicle position when the vehicle starts.
            # It is an ‘East-North-Up’ coordinate frame. ‘Up(z)’ is aligned with the gravity vector,
            # positive upwards. ‘East(x)’ points directly east along the line of latitude. ‘North(y)’
            # points towards the north pole.

            # Match vkitti: waymo global to vkitti global z --> -y, z --> y
            waymo2vkitti_world = np.array([[1., 0., 0., 0.],
                                           [0., 0., -1., 0.],
                                           [0., 1., 0., 0.],
                                           [0., 0., 0., 1.]])
            
            # Vehicle Frame
            # The x-axis is positive forwards, y-axis is positive to the left, z-axis is positive upwards.
            # A vehicle pose defines the transform from the vehicle frame to the global frame.


            # Vehicle to global frame transformation for this frame
            v2w_frame_i = np.reshape(frame.pose.transform, [4, 4])
            v2w_frame_i = np.matmul(waymo2vkitti_world, v2w_frame_i)

            # Get all objects in this frames
            frame_obj_dict = get_frame_objects(frame.laser_labels, v2w_frame_i, waymo_obj_meta, speed_thresh)

            # Loop over all camera images and visible objects per camera
            for camera_image in frame.images:
                for projected_lidar_labels in frame.projected_lidar_labels:
                    if projected_lidar_labels.name != camera_image.name or projected_lidar_labels.name not in cameras:
                        continue

                    for calibration in frame.context.camera_calibrations:
                        if calibration.name != camera_image.name:
                            continue

                        cam_no = np.array(camera_image.name).astype(np.float32)[None]
                        frame_no = np.array(i).astype(np.float32)[None]
                        count.append(len(images))

                        # Extract images and camera pose
                        images.append(np.array(tf.image.decode_jpeg(camera_image.image)))
                        extrinsic_transform_c2w = get_camera_pose(v2w_frame_i, calibration)
                        poses.append(extrinsic_transform_c2w)

                        # Extract dynamic objects for image
                        image_objects = np.ones([max_n_obj, 14]) * -1.
                        images_boxes = []
                        i_obj = 0
                        for label_2d in projected_lidar_labels.labels:
                            track_id = label_2d.id[:22]

                            # Only add objects with 3D information/dynamic objects
                            if track_id in frame_obj_dict:
                                pose_3d = frame_obj_dict[track_id]
                                dim = np.array(waymo_obj_meta[track_id][3]).astype(np.float32)
                                # Move vehicle reference point to bottom of the box like vkitti
                                pose_3d[1] = pose_3d[1] + (dim[1] / 2)

                                internal_track_id = np.array(waymo_obj_meta[track_id][1]).astype(np.float32)[None]
                                obj_type = np.array(waymo_obj_meta[track_id][2]).astype(np.float32)[None]

                                obj = np.concatenate([frame_no, cam_no, internal_track_id, obj_type, dim, pose_3d])

                                image_objects[i_obj, :] = obj
                                i_obj += 1

                                # Extract 2D bounding box for training
                                bbox_2d = get_bbox_2d(label_2d)
                                images_boxes.append(bbox_2d)

                        if i_obj > max_frame_obj:
                            max_frame_obj = i_obj

                        bboxes.append(images_boxes)
                        visible_objects.append(np.array(image_objects))

            if len(frames) >= max_frames:
                break

    if max_frame_obj > 0:
        visible_objects = np.array(visible_objects)[:, :max_frame_obj, :]
    else:
        print(max_frame_obj)
        print(visible_objects)
        visible_objects = np.array(visible_objects)[:, None, :]
    poses = np.array(poses)
    bboxes = np.array(bboxes)
    images = (np.maximum(np.minimum(np.array(images), 255), 0) / 255.).astype(np.float32)


    focal = np.reshape(frame.context.camera_calibrations[0].intrinsic, [9])[0]
    H = frame.context.camera_calibrations[0].height
    W = frame.context.camera_calibrations[0].width


    i_split = [np.sort(count[:]),
               count[int(0.8 * len(count)):],
               count[int(0.8 * len(count)):]]

    novel_view = 'left'
    n_oneside = int(poses.shape[0]/2)

    render_poses = poses[:1]
    # Novel view middle between both cameras:
    if novel_view == 'mid':
        new_poses_o = ((poses[n_oneside:, :, -1] - poses[:n_oneside, :, -1]) / 2) + poses[:n_oneside, :, -1]
        new_poses = np.concatenate([poses[:n_oneside, :, :-1], new_poses_o[...,None]], axis=2)
        render_poses = new_poses

    elif novel_view == 'left':
        # Render at trained left camera pose
        render_poses = poses[:n_oneside, ...]
    elif novel_view == 'right':
        # Render at trained left camera pose
        render_poses = poses[n_oneside:, ...]

    if use_obj:
        render_objects = visible_objects[:n_oneside, ...]
    else:
        render_objects = None

    # Create meta file matching vkitti2 meta data
    objects_meta = {}
    for meta_value in waymo_obj_meta.values():
        objects_meta[meta_value[1]] = np.concatenate([np.array(meta_value[1])[None],
                                                      meta_value[3],
                                                      np.array([meta_value[2]]) ])

    half_res = True
    if half_res:
        print('Using half resolution!!!')
        H = H // 2
        W = W // 2
        focal = focal / 2.
        images = tf.image.resize_area(images, [H, W]).numpy()

        for frame_boxes in bboxes:
            for i_box, box in enumerate(frame_boxes):
                frame_boxes[i_box] = box // 2


    return images, poses, render_poses, [H, W, focal], i_split, visible_objects, objects_meta, render_objects, bboxes, waymo_obj_meta