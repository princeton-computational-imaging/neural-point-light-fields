import os

import open3d as o3d
import numpy as np
import torch
import torch.nn as nn

class ICP(nn.Module):
    def __init__(self, n_frames=5, step_width=2):
        super(ICP, self).__init__()
        self.n_frames = n_frames
        self.step_width = step_width

        self._merged_pcd_cache = {}
        # nppcd = np.load('./src/pointLF/icp/all_pts.npy')  #id: 8,0,4,6,10,8,6,6
        # poses = np.load('./src/pointLF/icp/lidar_poses.npy') #id: 0,1,2,3,4,5,6,7,8,9,10
        #
        #
        # pts_4 = self.array2pcd(self.transformpts(nppcd[2], poses[4]))
        # pts_6 = self.array2pcd(self.transformpts(nppcd[3], poses[6]))
        # pts_10 = self.array2pcd(self.transformpts(nppcd[4], poses[10]))
        #
        # pts_4.paint_uniform_color([1, 0.706, 0])
        # pts_6.paint_uniform_color([0, 0.651, 0.929])
        # pts_10.paint_uniform_color([0.0, 1.0, 0.0])
        #
        # pts_4 = self.icp_core(pts_4,pts_10)
        # pts_6 = self.icp_core(pts_6,pts_10)
        # o3d.visualization.draw_geometries([pts_4,pts_6,pts_10])

    def forward(self, scene, cam_frame_id, caching=False, augment=False, pc_frame_id=None):
        '''

        :param scene:
        :type scene:
        :param cam_frame_id:
        :type cam_frame_id:
        :param caching:
        :type caching:
        :param augment: Do not choose the pcd from the current frame but a frame in front or after
        :type augment:
        :return:
        :rtype:
        '''
        li_sel_idx = np.array([v.scene_idx if v.name == 'TOP' else -1 for k, v in scene.nodes['lidar'].items()])
        top_lidar_id = li_sel_idx[np.where(li_sel_idx > 0)][0]

        if pc_frame_id is None:
            if augment:
                max_augment_distance = 5
                augment_distance = np.random.randint(0, max_augment_distance * 2 + 1) - max_augment_distance
                pc_frame_id = np.minimum(np.maximum(cam_frame_id + augment_distance, 0), len(scene.frames) - 1)
            else:
                pc_frame_id = cam_frame_id

        current_points_frame = scene.frames[pc_frame_id]
        current_camera_frame = scene.frames[cam_frame_id]

        pcd_path = scene.frames[pc_frame_id].point_cloud_pth[top_lidar_id]

        if not scene.scene_descriptor["type"] == "kitti":
            if scene.scene_descriptor.get('pt_cloud_fix', False):
                merged_pcd_dir = os.path.join(
                    *(
                            ["/"] +
                            pcd_path.split("/")[1:-1] +
                            ['merged_pcd_full_{}_scene_{}_{}_frames_{}_{}_n_fr_{}'.format(
                                scene.scene_descriptor['type'],
                                # str(scene.scene_descriptor['scene_id']).zfill(4),
                                str(scene.scene_descriptor['scene_id'][0]).zfill(4),
                                str(scene.scene_descriptor['scene_id'][1]).zfill(4),
                                str(scene.scene_descriptor['first_frame']).zfill(4),
                                str(scene.scene_descriptor['last_frame']).zfill(4),
                                str(self.n_frames).zfill(4) if self.n_frames is not None else str('all'),)
                            ]
                    )
                )
            else:
                merged_pcd_dir = os.path.join(
                    *(
                            ["/"] +
                            pcd_path.split("/")[1:-1] +
                            ['merged_pcd_{}_scene_{}_{}_frames_{}_{}_n_fr_{}'.format(
                                scene.scene_descriptor['type'],
                                # str(scene.scene_descriptor['scene_id']).zfill(4),
                                str(scene.scene_descriptor['scene_id'][0]).zfill(4),
                                str(scene.scene_descriptor['scene_id'][1]).zfill(4),
                                str(scene.scene_descriptor['first_frame']).zfill(4),
                                str(scene.scene_descriptor['last_frame']).zfill(4),
                                str(self.n_frames).zfill(4) if self.n_frames is not None else str('all'), )
                            ]
                    )
                )
        else:
            merged_pcd_dir = os.path.join(
                *(
                        ["/"] +
                        pcd_path.split("/")[1:-1] +
                        ['merged_pcd_full_{}_scene_{}_frames_{}_{}_n_fr_{}'.format(
                            scene.scene_descriptor['type'],
                            str(scene.scene_descriptor['scene_id']).zfill(4),
                            str(scene.scene_descriptor['first_frame']).zfill(4),
                            str(scene.scene_descriptor['last_frame']).zfill(4),
                            str(self.n_frames).zfill(4) if self.n_frames is not None else str('all'), )
                        ]
                )
            )

        merged_pcd_path = os.path.join(
            merged_pcd_dir, '{}.pcd'.format(str(pc_frame_id).zfill(6))
        )

        os.umask(0)
        if not os.path.isdir(merged_pcd_dir):
            os.mkdir(merged_pcd_dir)
        else:
            # TODO: Add version check here
            if os.path.isfile(merged_pcd_path):
                current_points_frame.merged_pcd_pth = merged_pcd_path

        # Get the transformation from world coordinates to the vehicle coordinates of the requested frame
        veh2wo_0 = current_points_frame.global_transformation
        wo2veh_0 = np.concatenate(
            [veh2wo_0[:3, :3].T,
             veh2wo_0[:3, :3].T.dot(-veh2wo_0[:3, 3])[:, None]],
            axis=1
        )
        wo2veh_0 = np.concatenate([wo2veh_0, np.array([[0., 0., 0., 1.]])])
        pose_0 = np.eye(4)

        # Get the transformation from the vehicle pose of the camera to the vehivle pose of the point cloud
        veh2wo_cam = current_camera_frame.global_transformation
        camera_trafo = wo2veh_0.dot(veh2wo_cam)

        if current_points_frame.merged_pcd_pth is None or (current_points_frame.merged_pcd is None and caching):
            all_points_post = None
            all_points_pre = None

            # Get points
            pts_0 = current_points_frame.load_point_cloud(top_lidar_id)
            pts_0 = self.array2pcd(self.transformpts(pts_0[:, :3], pose_0))

            all_points = pts_0
            all_points, _ = all_points.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

            first_fr_id = min(scene.frames.keys())
            if self.n_frames is not None:
                pre_current = np.linspace(cam_frame_id - 1, first_fr_id, cam_frame_id, dtype=int)[:self.n_frames]
            else:
                pre_current = np.linspace(cam_frame_id - 1, first_fr_id, cam_frame_id, dtype=int)

            last_fr_id = max(scene.frames.keys())
            if self.n_frames is not None:
                post_current = np.linspace(cam_frame_id + 1, last_fr_id, last_fr_id - cam_frame_id, dtype=int)[:self.n_frames]
            else:
                post_current = np.linspace(cam_frame_id + 1, last_fr_id, last_fr_id - cam_frame_id, dtype=int)

            # Loop over adjacent frames in the future
            # all_points_post = self.merge_adajcent_points(post_current, scene, wo2veh_0, pts_0, top_lidar_id)

            for fr_id in np.concatenate([post_current]):
                frame_i = scene.frames[fr_id]
                # Load point cloud
                pts_i = frame_i.load_point_cloud(top_lidar_id)

                # Do not keep dynamic scene parts behind the geo vehicle from future frames
                pts_front_idx = np.where(pts_i[:, 0] > 0.)
                pts_back_idx = np.where(np.all(np.stack([pts_i[:, 0] < 0., np.abs(pts_i[:, 1]) > 1.5]), axis=0))
                pts_idx = np.concatenate([pts_front_idx[0], pts_back_idx[0]])
                pts_i = pts_i[pts_idx]

                # Get Transformation from veh frame to world
                veh2wo_i = frame_i.global_transformation

                # Center all point clouds at the requested frame
                # Waymo
                pose_i = wo2veh_0.dot(veh2wo_i)

                # Transform point cloud into the vehicle frame of the current frame
                pts_i = self.array2pcd(self.transformpts(pts_i[:, :3], pose_i))
                # Remove noise from point cloud
                pts_i, _ = pts_i.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

                if all_points_post is None:
                    # Match first pointcloud close to the selected frames lidar pose
                    pts_i = self.icp_core(pts_i, pts_0)
                    all_points_post = pts_i
                else:
                    pts_i = self.icp_core(pts_i, all_points_post)
                    all_points_post = all_points_post + pts_i

            if all_points_post is not None:
                all_points_post = self.icp_core(all_points_post, pts_0)
                all_points += all_points_post

            # Loop over adjacent frames in the past
            # all_points_pre = self.merge_adajcent_points(pre_current, scene, wo2veh_0, pts_0, top_lidar_id)

            for fr_id in np.concatenate([pre_current]):
                frame_i = scene.frames[fr_id]
                # Load point cloud
                pts_i = frame_i.load_point_cloud(top_lidar_id)

                # Get Transformation from veh frame to world
                veh2wo_i = frame_i.global_transformation

                # Center all point clouds at the requested frame
                # Waymo
                pose_i = wo2veh_0.dot(veh2wo_i)

                # Transform point cloud into the vehicle frame of the current frame
                pts_i = self.array2pcd(self.transformpts(pts_i[:, :3], pose_i))
                # Remove noise from point cloud
                pts_i, _ = pts_i.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

                if all_points_pre is None:
                    # Match first pointcloud close to the selected frames lidar pose
                    pts_i = self.icp_core(pts_i, pts_0)
                    all_points_pre = pts_i
                else:
                    pts_i = self.icp_core(pts_i, all_points_pre)
                    all_points_pre = all_points_pre + pts_i

            if all_points_pre is not None:
                all_points_pre = self.icp_core(all_points_pre, pts_0)
                all_points  += all_points_pre

            # Outlier and noise removal
            all_points, ind = all_points.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

            if caching:
                print("Caching pcd")
                current_points_frame.merged_pcd = all_points
            else:
                # Store merged point cloud for future readings
                o3d.io.write_point_cloud(merged_pcd_path, all_points)
        else:
            if caching:
                print("Retriving pcd from cache")
                all_points = current_points_frame.merged_pcd
            else:
                # t0 = time.time()
                all_points = o3d.io.read_point_cloud(current_points_frame.merged_pcd_pth, format='pcd')
                # print("Read points {}".format(time.time()- t0))

        pts = np.asarray(all_points.points)

        return pts, merged_pcd_path, camera_trafo, pc_frame_id


    def merge_adajcent_points(self, fr_id_ls, scene, wo2veh_0, pts_0, top_lidar_id):

        # Loop over adjacent frames in the future
        for fr_id in np.concatenate([fr_id_ls]):
            frame_i = scene.frames[fr_id]
            # Load point cloud
            pts_i = frame_i.load_point_cloud(top_lidar_id)

            # Do not keep dynamic scene parts hiding behind the ego vehicle from adjacent frames
            pts_front_idx = np.where(pts_i[:, 0] > 0.)
            pts_back_idx = np.where(np.all(np.stack([pts_i[:, 0] < 0., np.abs(pts_i[:, 1]) > 1.5]), axis=0))
            pts_idx = np.concatenate([pts_front_idx[0], pts_back_idx[0]])
            pts_i = pts_i[pts_idx]

            # Get Transformation from veh frame to world
            veh2wo_i = frame_i.global_transformation

            # Center all point clouds at the requested frame
            # Waymo
            pose_i = wo2veh_0.dot(veh2wo_i)

            # Transform point cloud into the vehicle frame of the current frame
            pts_i = self.array2pcd(self.transformpts(pts_i[:, :3], pose_i))
            # Remove noise from point cloud
            pts_i, _ = pts_i.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

            if all_points_adj is None:
                # Match first pointcloud close to the selected frames lidar pose
                pts_i = self.icp_core(pts_i, pts_0)
                all_points_adj = pts_i
            else:
                pts_i = self.icp_core(pts_i, all_points_adj)
                all_points_adj = all_points_adj + pts_i

        return all_points_adj


    def transformpts(self,pts,pose):
        pts = np.concatenate((pts,np.ones((pts.shape[0],1))),1)
        pts = (pose.dot(pts.T)).T
        return pts[:,:3]

    def array2pcd(self, all_pts, color=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts)
        if color is not None:
            pcd.paint_uniform_color(color)
        return pcd

    def icp_core(self,processed_source,processed_target):
        threshold = 1.0
        trans_init = np.eye(4).astype(np.int)

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source=processed_source,
            target=processed_target,
            max_correspondence_distance=threshold,
            init=trans_init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        # print(reg_p2p.transformation)
        processed_source.transform(reg_p2p.transformation)
        return processed_source
