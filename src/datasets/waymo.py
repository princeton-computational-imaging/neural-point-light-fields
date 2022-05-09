import os
import pickle
import numpy as np
from collections import defaultdict
from copy import deepcopy

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import label_pb2 as open_label

# from . import dataset_pb2 as open_dataset
# from . import label_pb2 as open_label
# from utils import invert_transformation,roty_matrix, rotz_matrix
from src.datasets.utils import invert_transformation,roty_matrix, rotz_matrix
import matplotlib.pyplot as plt
from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles
import torch
# waymo2vkitti_transform = np.array([[0., -1., 0., 0.],
#                                    [-1., 0., 0., 0.],
#                                    [0., 0., -1., 0.],
#                                    [0., 0., 0., 1.]])


waymo2vkitti_transform = np.array([[1., 0., 0., 0.],
                                   [0., 0., 1., 0.],
                                   [0., 1., 0., 0.],
                                   [0., 0., 0., 1.]])
waymo2kitti_type = dict(zip(open_label.Label.Type.values(),[-1,0,4,-1,-1]))

class Waymo:
    def __init__(self, datadir, scene_dict, selected_frames,object_types=None):
        USE_PER_CAMERA_VEH_POSE = True
        self.num_cameras = len(open_dataset.CameraName.Name.values())-1
        self.num_lasers = len(open_dataset.LaserName.Name.values()) - 1
        with open(os.path.join(datadir,'tracking_info.pkl'),'rb') as f:
            tracking_info = pickle.load(f)

        new_dict,cam_params = {},{}
        non_frame_keys = [key for key in tracking_info.keys() if isinstance(key,str)]
        for key in non_frame_keys:
            cam_params[key] = tracking_info.pop(key)

        if selected_frames is None:
            selected_frames = [0, len(tracking_info.keys())]
        # self.debug_tracking_info(tracking_info)
        
            
        self.global_frame_ID_enum = {}
        for ind,key in enumerate(sorted(list(tracking_info.keys()))[selected_frames[0]:selected_frames[1]+1]):
            new_dict[key] = deepcopy(tracking_info[key])
            self.global_frame_ID_enum[key] = ind+selected_frames[0]
        tracking_info = deepcopy(new_dict)

        self.images = [tracking_info[k]['im_paths'][cam_ind] for cam_ind in
         range(1,len(open_dataset.CameraName.Name.values())) for k in tracking_info.keys()
          ]
        self.point_cloud_pth = [tracking_info[k]['pcd_paths'][li_ind] for li_ind in
            range(1, len(open_dataset.LaserName.Name.values())) for k in tracking_info.keys()
        ]
        # cam_params = {'intrinsic':tracking_info['intrinsic']}
        self.num_cam_frames = len(self.images)
        assert np.round(len(self.images)/self.num_cameras)==len(self.images)/self.num_cameras
        self.num_frames = len(self.images)//self.num_cameras
        # Faulty code, assuming outer loop is over frames:
        # self.frame_cam_enum = lambda frame_tuple:frame_tuple[0]*self.num_cameras+frame_tuple[1]-1
        # Fixed code, assuming outer loop is over cameras:
        self.frame_cam_enum = lambda frame_tuple:(frame_tuple[1]-1)*self.num_frames+frame_tuple[0]
        self.focal = dict(zip(open_dataset.CameraName.Name.keys()[1:],
                    [read_intrinsic(cam_params['intrinsic'][i])['f_u']
                     for i in range(self.num_cameras)]))
        self.H = dict(zip(open_dataset.CameraName.Name.keys()[1:],[cam_params['height'][i] for i in range(self.num_cameras)]))
        self.W = dict(zip(open_dataset.CameraName.Name.keys()[1:], [cam_params['width'][i] for i in range(self.num_cameras)]))
        self.hwf = {'focal':self.focal,'height':self.H,'width':self.W}
        self.laser_name = open_dataset.LaserName.Name.keys()[1:]
        object_types = [open_label.Label.Type.Value(name) for name in object_types]
        valid_obj_type = lambda obj_data: obj_data['type'] in object_types
        self.visible_objects = defaultdict(dict)
        self.objects_meta = {}
        for f_counter,(f_num,frame) in enumerate(tracking_info.items()):
            for cam_int,camera_labels in frame['camera_labels'].items():
                for id,obj_data_2D in camera_labels.items():
                    if not valid_obj_type(obj_data_2D):    continue
                    if id not in self.objects_meta:
                        self.objects_meta[id] = {'type':obj_data_2D['type']}
                        for key in ['length','width','height']:
                            self.objects_meta[id][key] = tracking_info[f_num]['lidar_labels'][id][key]
                    self.visible_objects[(f_counter, cam_int)][id] = self.obj_pose_in_world(tracking_info[f_num]['lidar_labels'][id],frame['veh_pose'])
                    self.visible_objects[(f_counter, cam_int)][id]['frame_id'] = f_num
        assert len(self.objects_meta)>0,'No relevant objects in the specified scene and frame range'
        self.obj_ID_enum = dict(zip([id for id in self.objects_meta.keys()],[i for i in range(1,len(self.objects_meta)+1)]))
        self.enum2objID = {v:k for k,v in self.obj_ID_enum.items()}
        self.scene_classes = np.unique([obj['type'] for obj in self.objects_meta.values()])
        self.scene_objects = [obj for obj in self.objects_meta.keys()]
        # all needed attributes
        self.scene_id = scene_dict['scene_id']
        self.type = scene_dict['type']
        self.box_scale = scene_dict['box_scale']

        # Object Data
        self.obj_poses = self.kitti_style_obj_properties()
        self.obj_meta_tensor = self.kitti_style_object_meta_tensor()
        self.add_input_rows = 2 # Coppied as is from the Kitti pipeline

        # scene data
        self.object_positions = self.kitti_style_object_position()
        self.near_plane = scene_dict['near_plane']
        self.far_plane = scene_dict['far_plane']

        n_frames = len(self.images) // 5
        # Get camera and vehicle poses:
        # Camera calibrations
        veh2cam = self.read_data_in_order(tracking_info, 'cam2veh')
        cam2veh_data = np.stack(veh2cam)

        # Laser calibrations
        veh2laser = self.read_data_in_order(tracking_info, 'veh2laser', n_sensors=self.num_lasers)
        laser2veh = np.stack(veh2laser)

        # Capture vehicle pose
        veh2world= self.repeat_and_read_data_in_order(tracking_info, 'veh_pose')
        world2veh = [invert_transformation(v[:3, :3], v[:3, 3]) for v in veh2world]

        # Capture vehicle pose at specific time
        veh2world_per_cam = self.read_data_in_order(tracking_info, 'per_cam_veh_pose')
        world2veh_per_cam = np.stack([invert_transformation(v[:3, :3], v[:3, 3]) for v in veh2world_per_cam])

        # Unused variables
        # world2veh = [invert_transformation(v[:3, :3], v[:3, 3]) for v in veh2world_debug]
        # cam2veh_true = [invert_transformation(v[:3, :3], v[:3, 3]) for v in veh2cam]
        # world2cam = np.stack([np.matmul(veh2cam[i], world2veh_per_cam[i]) for i in range(self.num_cam_frames)], 0)

        # Adjust calibrated camera pose for vehicle movement
        veh2cam = np.matmul(world2veh_per_cam, np.matmul(np.stack(veh2world), cam2veh_data))
        # cam_diff_trans = np.stack(veh2world_debug)[:, :3, 3] - np.stack(veh2world_per_cam)[:, :3, 3]
        # veh2cam[:, :3, 3] = [v[:3, 3] - np.matmul(world2veh[i][:3, :3], cam_diff_trans[i]) for i, v in enumerate(veh2cam)]
        #
        # # Plot all sensors + Vehicle in World coordinates
        # # TODO: Test for Side Left etc.
        # plt.figure()
        # plt.axis('equal')
        # plt.scatter(np.stack(veh2world_per_cam)[:n_frames, :3, 3].reshape(-1, 3)[:, 0],
        #             np.stack(veh2world_per_cam)[:n_frames, :3, 3].reshape(-1, 3)[:, 1], c="cyan")
        # plt.scatter(np.stack(veh2world)[:n_frames, :3, 3].reshape(-1, 3)[:, 0],
        #             np.stack(veh2world)[:n_frames, :3, 3].reshape(-1, 3)[:, 1], c="blue")
        # unaligned_cam_poses = np.matmul(veh2world, veh2cam)[:n_frames, :3, 3]
        # # aligned_cam_poses = np.matmul(veh2world_per_cam, veh2cam)[:n_frames, :3, 3]
        # lidar_poses = np.matmul(veh2world, veh2laser)[:n_frames, :3, 3]
        #
        # plt.scatter(unaligned_cam_poses[:, 0], unaligned_cam_poses[:, 1], c="green")
        # # plt.scatter(aligned_cam_poses[:, 0], aligned_cam_poses[:, 1])
        # plt.scatter(lidar_poses[:, 0], lidar_poses[:, 1], c="orange")
        #
        # norm_dist_laser_cam = np.linalg.norm(unaligned_cam_poses - lidar_poses, axis=1)

        # Convert camera pose axis to KITTI Style
        cam_poses = [np.matmul(pose, np.array([[0., 0., 1., 0.],
                                             [1., 0., 0., 0.],
                                             [0., 1., 0., 0.],
                                             [0., 0., 0., 1.], ])) for pose in veh2cam]


        cam_poses = np.stack([[cam_poses[i * n_frames + j] for j in range(n_frames)] for i in range(self.num_cameras)])

        self.poses = cam_poses.reshape(n_frames * self.num_cameras, 4, 4)

        # Output world pose of camera
        # world2cam = np.matmul(veh2world_per_cam, veh2cam)
        # cam2world = np.matmul(veh2world, veh2cam)
        cam2world = np.matmul(veh2world_per_cam, cam2veh_data)

        cam_poses_world = [np.matmul(pose, np.array([[0., 0., 1., 0.],
                                                     [1., 0., 0., 0.],
                                                     [0., 1., 0., 0.],
                                                     [0., 0., 0., 1.], ])) for pose in cam2world]

        cam_poses_world = np.stack([[cam_poses_world[i * n_frames + j] for j in range(n_frames)] for i in range(self.num_cameras)])

        self.poses_world = cam_poses_world.reshape(n_frames * self.num_cameras, 4, 4)

        # Output Lidar poses
        laser_poses = laser2veh
        laser_poses = np.stack([[laser_poses[i * n_frames + j] for j in range(n_frames)] for i in range(self.num_lasers)])

        lidar_poses_world = np.matmul(veh2world, laser2veh)
        lidar_poses_world = np.stack([[lidar_poses_world[i * n_frames + j] for j in range(n_frames)] for i in range(self.num_lasers)])

        self.lidar_poses = laser_poses.reshape(n_frames * self.num_lasers, 4, 4)
        self.lidar_poses_world = lidar_poses_world.reshape(n_frames * self.num_lasers, 4, 4)

        self.veh_pose = np.stack(veh2world)[:n_frames]

        self.obj_poses[..., 3] = self.obj_poses[..., 3] # - (np.pi / 2)

        self.convert_data_2_kitti_style()

        ######### DEBUG ############
        # self.debug_outputs()
        print('Loaded Waymo')

    def convert_data_2_kitti_style(self):
        #     Converting the visible_objects and objects_meta dictionaries to the Kitti pipeline convention:
        visible_objects = -1*np.ones([self.num_cam_frames,self.max_input_objects,14])
        for frame_cam, objects in self.visible_objects.items():
            for obj_num, (obj_ID, obj_data) in enumerate(objects.items()):
                visible_objects[self.frame_cam_enum(frame_cam),obj_num] =\
                    np.array([self.global_frame_ID_enum[obj_data['frame_id']],frame_cam[1],self.obj_ID_enum[obj_ID],0,self.objects_meta[obj_ID]['length']
                              ,self.objects_meta[obj_ID]['height'],self.objects_meta[obj_ID]['width'],
                              # obj_data['c_y'], obj_data['c_z'], obj_data['c_x'], obj_data['heading'], 0, 0, 1])
                              obj_data['c_x'],obj_data['c_y'],obj_data['c_z'],obj_data['heading'],0,0,1])
        objects_meta = {}
        for obj_ID, obj_data in self.objects_meta.items():
            objects_meta[self.obj_ID_enum[obj_ID]] = np.array([self.obj_ID_enum[obj_ID],obj_data['length'],obj_data['height'],obj_data['width'],obj_data['type']])
        self.visible_objects,self.objects_meta = visible_objects,objects_meta


    def kitti_style_object_position(self):
        object_positions = -1*np.ones([self.num_cam_frames*self.max_input_objects,3])
        for frame_cam,objects in self.visible_objects.items():
            for obj_num,obj_data in enumerate(objects.values()):
                object_positions[self.frame_cam_enum(frame_cam)*self.max_input_objects+obj_num] = read_coords_vect(obj_data)
        return object_positions

    def kitti_style_object_meta_tensor(self):
        obj_meta_tensor = [np.array([-1,0,0,0,0])]
        for obj_ID,obj_data in self.objects_meta.items():
            dims_factor = 1.2 if open_label.Label.Type.Name(obj_data['type'])=='TYPE_PEDESTRIAN' else self.box_scale
            obj_meta_tensor.append(np.array([self.obj_ID_enum[obj_ID],dims_factor*obj_data['length'],obj_data['height'],dims_factor*obj_data['width'],obj_data['type']]))
        return np.stack(obj_meta_tensor,0)

    def kitti_style_obj_properties(self):
        self.max_input_objects = max([len(objs.keys()) for objs in self.visible_objects.values()])
        obj_properties = -1*np.ones([self.num_cam_frames,self.max_input_objects,6]) # Repeating the last property of enumerated obejct ID twice, since the actual object ID is a string in the Waymo case.
        for frame_cam,objects in self.visible_objects.items():
            for obj_num,(obj_ID,obj_data) in enumerate(objects.items()):
                # obj_properties[self.frame_cam_enum(frame_cam),obj_num,:3] = read_coords_vect(obj_data)[[1, 2, 0]]
                obj_properties[self.frame_cam_enum(frame_cam), obj_num, :3] = read_coords_vect(obj_data)[[0, 1, 2]]

                obj_properties[self.frame_cam_enum(frame_cam),obj_num,3:] = np.stack([obj_data['heading']]+2*[self.obj_ID_enum[obj_ID]])
        return obj_properties


    def read_data_in_order(self,tracking_info, field, n_sensors=None):
        n_sensors = n_sensors if n_sensors is not None else self.num_cameras
        return [ti[field][cam_ind] for cam_ind in range(n_sensors) for ti in tracking_info.values()]

    def repeat_and_read_data_in_order(self, tracking_info, field, n_sensors=None):
        n_sensors = n_sensors if n_sensors is not None else self.num_cameras
        return [ti[field] for cam_ind in range(n_sensors) for ti in tracking_info.values()]

    def obj_pose_in_world(self,obj_data,veh_pose):
        obj_pos = np.eye(4)
        obj_pos[:3,3] = read_coords_vect(obj_data)
        obj_pos[:3, :3] = rotz_matrix(obj_data['heading'])
        # obj_pos = np.dot(veh_pose, obj_pos)
        heading_angle = -np.arctan2(obj_pos[1, 0], obj_pos[0, 0]) # - np.pi /2
        if np.abs(heading_angle) > np.pi:
            if heading_angle < 0.:
                heading_angle = np.pi + (heading_angle + np.pi)
            else:
                heading_angle = -np.pi + (heading_angle - np.pi)
        # return {'c_x':obj_pos[0,3] - 3607.,'c_y':obj_pos[1,3] - 1975.,'c_z':obj_pos[2,3] + 190.,'heading':-np.arctan2(obj_pos[1, 0], obj_pos[0, 0])+3.*np.pi/2}
        return {'c_x': obj_pos[0, 3], 'c_y': obj_pos[1, 3], 'c_z': obj_pos[2, 3],'heading': heading_angle}

    def debug_tracking_info(self, tracking_info):
        veh_poses = []
        cam2veh_calib = []
        per_cam_veh_pose = []
        n_iter = len(tracking_info) - 10
        for k in range(n_iter):
            veh_poses.append(tracking_info[(0, k)]['veh_pose'])
            cam2veh_calib.append(tracking_info[(0, k)]['cam2veh'])
            per_cam_veh_pose.append(tracking_info[(0, 0)]['per_cam_veh_pose'])

        veh_poses = np.stack(veh_poses)
        cam2veh_calib = np.stack(cam2veh_calib)
        veh_translation = veh_poses[:, :3, 3]
        plt.figure()
        plt.scatter(veh_translation[:, 0], veh_translation[:, 1])
        plt.axis('equal')
        plt.show()

    def debug_outputs(self):
        n_frames = len(self.images) // 5
        all_obj_poses = self.obj_poses.reshape(5, -1, self.obj_poses.shape[1], 6)
        cam_poses = np.stack([[self.poses[i * n_frames + j] for j in range(n_frames)] for i in range(5)])

        for s in [1, 2]:
            for i in range(n_frames):
                plt_obj = all_obj_poses[:, i].reshape(-1, 6)
                obj_pos = plt_obj[:, :3]
                obj_yaw = plt_obj[:, 3]

                cam_trans = cam_poses[:, i, :3, 3]
                plt_cam_trans = cam_trans.reshape(-1, 3)
                plt_cam_rot = cam_poses[:, i, :3, :3].reshape(-1, 3, 3)
                plt.figure()
                plt.scatter(obj_pos[:, 0], obj_pos[:, s])
                plt.scatter(plt_cam_trans[:, 0], plt_cam_trans[:, s])
                plt.axis('equal')
                for l in range(5):
                    for j in range(3):
                        c = [0, 0, 0]
                        c[j] = 1
                        arr = plt_cam_rot[l, :, j]
                        plt.arrow(cam_trans[l, 0], cam_trans[l, s], arr[0], arr[s], color=c)

                for o, (pos, yaw) in enumerate(zip(obj_pos, obj_yaw)):
                    for j in range(1):
                        rot_mat = euler_angles_to_matrix(torch.tensor([0., 0., yaw]), 'XYZ').cpu().numpy()
                        c = [0, 0, 0]
                        c[j] = 1
                        arr = rot_mat[:, j]
                        plt.arrow(pos[0], pos[s], arr[0], arr[s], color=c)

def read_coords_vect(obj_data):
    return np.array([obj_data['c_x'],obj_data['c_y'],obj_data['c_z']])

def read_intrinsic(intrinsic_params_vector):
    return dict(zip(['f_u', 'f_v', 'c_u', 'c_v', 'k_1', 'k_2', 'p_1', 'p_2', 'k_3'],intrinsic_params_vector))
