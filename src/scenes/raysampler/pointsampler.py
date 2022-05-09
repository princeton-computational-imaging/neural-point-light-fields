import numpy as np
import torch
import torch.nn as nn
import open3d as o3d

from pytorch3d.renderer.implicit.utils import RayBundle
from .rayintersection import RayPlaneIntersection, RayBoxIntersection
from pytorch3d.transforms import Transform3d, Translate, Rotate, Scale
from .frustum_helpers import get_frustum, get_frustum_torch

from src.pointLF.icp.pts_registration import ICP

import matplotlib.pyplot as plt


class PointDistanceRaysamplerTorch(nn.Module):

    def __init__(self,
                 n_pts: int,
                 k_closest: int,
                 point_chunk_size: int = 1e12,
                 merge_pcd: bool = False,
                 n_frames_merged=5,
                 augment_point_cloud=False,
                 new_encoding=False,
                 pt_caching=False,
                 ):

        super(PointDistanceRaysamplerTorch, self).__init__()

        self.n_pts = n_pts
        if k_closest == 0:
            k_closest = 2
        self.k_closest = k_closest
        self._pt_chunk_sz = point_chunk_size
        self.merge_pcd = merge_pcd
        if merge_pcd:
            self.icp = ICP(n_frames=n_frames_merged, )
            self.augment = augment_point_cloud
        self._pt_caching = pt_caching
        self._pts_dict = {}

        self.new_enc = new_encoding

    def forward(self,
                ray_bundle: RayBundle,
                scene,
                validation, ):
        device = ray_bundle.origins.device

        # Prepare ray_bundle.lengths to keep the point projection along each ray
        if not self.new_enc:
            ray_bundle = ray_bundle._replace(
                lengths=torch.zeros([ray_bundle.lengths.shape[0], ray_bundle.lengths.shape[1], self.k_closest],
                                    device=device))
        xycfn = ray_bundle.xys.cpu().numpy()

        # Get camera and frame combinations
        c_fr_idx = ray_bundle.xys[..., [2, 3]].unique(dim=1)[0]
        fr_idx = ray_bundle.xys[..., 3].unique(dim=1)[0]

        lidar_idx = list(scene.nodes['lidar'])
        lidar = scene.nodes['lidar'][lidar_idx[0]]

        closest_point_mask = {}
        closest_point_dist = {}
        closest_point_azimuth = {}
        closest_point_pitch = {}
        pt_cloud_select = {}
        pt_cloud_select_new = {}
        ray_dirs = {}

        # Iterate over frames
        for f in fr_idx:
            f = int(f)
            try:
                frame = scene.frames[int(f)]
            except:
                KeyError('Frame with the ID {} does not exist for this scene'.format(int(f)))

            # Select all rays from the same frame
            if ray_bundle.xys.dim() >= 4:
                frame_mask = np.where(xycfn[..., 0, 3] == f)
            else:
                frame_mask = np.where(xycfn[..., 3] == f)

            # Relevant cameras and their edges in the scene graph of the current frame
            cam_idx = np.unique(xycfn[frame_mask][..., 2])
            cam_nodes = [scene.nodes['camera'][int(c)] for c in cam_idx]
            cam_ed_ls = [scene.frames[int(f)].get_edge_by_child_idx([int(c)])[0][0] for c in cam_idx]

            # Get ray origins and ray directions for these rays
            ray_o = ray_bundle.origins
            ray_d = ray_bundle.directions

            li_sel_idx = np.array([v.scene_idx if v.name == 'TOP' else -1 for k, v in scene.nodes['lidar'].items()])
            top_lidar_id = li_sel_idx[np.where(li_sel_idx > 0)][0]
            # TODO: Add point cloud loading and icp to get_item of the scene similar to image pass
            if not self.merge_pcd:
                # Use point cloud of this frame
                pt_cloud_li = scene.frames[int(f)].load_point_cloud(top_lidar_id)[..., :3]
                # pt_cloud_li = pt_cloud_li[np.where(pt_cloud_li[..., 0] >= 0.)]
                pc_frame_id = int(f)

            else:
                augment = self.augment and not validation
                # Use point cloud merged across frames
                if not self._pt_caching:
                    pt_cloud_li, _, camera_trafo, pc_frame_id = self.icp(scene, cam_frame_id=int(f), augment=augment)
                else:

                    pt_cloud_li = []
                    if augment:
                        max_augment_distance = 5
                        augment_distance = np.random.randint(0, max_augment_distance * 2 + 1) - max_augment_distance
                        pc_frame_id = np.minimum(np.maximum(int(f) + augment_distance, 0),
                                                 len(scene.frames) - 1)
                    else:
                        pc_frame_id = int(f)

                    if not pc_frame_id in self._pts_dict:
                        pt_cloud_li, _, camera_trafo, pc_frame_id = self.icp(scene, cam_frame_id=int(f), augment=augment,
                                                                             pc_frame_id=pc_frame_id)
                    elif augment:
                        # Get the transformation from world coordinates to the vehicle coordinates of the requested frame
                        veh2wo_0 = scene.frames[pc_frame_id].global_transformation
                        wo2veh_0 = np.concatenate(
                            [veh2wo_0[:3, :3].T,
                             veh2wo_0[:3, :3].T.dot(-veh2wo_0[:3, 3])[:, None]],
                            axis=1
                        )
                        wo2veh_0 = np.concatenate([wo2veh_0, np.array([[0., 0., 0., 1.]])])
                        pose_0 = np.eye(4)

                        # Get the transformation from the vehicle pose of the camera to the vehivle pose of the point cloud
                        veh2wo_cam = scene.frames[int(f)].global_transformation
                        camera_trafo = wo2veh_0.dot(veh2wo_cam)

                if augment:
                    # change camera according to the augmentad new point cloud frame
                    ray_o = torch.matmul(torch.tensor(camera_trafo, device=ray_o.device, dtype=ray_o.dtype),
                                         torch.cat([ray_o, torch.ones_like(ray_o)[..., :1]], axis=-1).squeeze().T).T[
                            None, ..., :3]
                    ray_d = torch.matmul(torch.tensor(camera_trafo, device=ray_d.device, dtype=ray_d.dtype)[:3, :3],
                                         ray_d.squeeze().T).T[None]

            # Get camera poses in local point cloud corrdinates and the camera viewing frustum
            veh2wo = torch.Tensor(scene.frames[pc_frame_id].global_transformation)
            wo2veh = torch.cat(
                [veh2wo[:3, :3].T,
                 torch.matmul(veh2wo[:3, :3].T, -veh2wo[:3, 3, None])],
                dim=1
            )

            wo2veh = torch.cat([wo2veh, torch.Tensor([[0., 0., 0., 1.]])])

            cam_mask_ls = [np.where(xycfn[frame_mask][..., 2] == int(c)) for c in cam_idx]
            cam_frame_mask_ls = [tuple([v[cam_mask] for v in frame_mask]) for cam_mask in cam_mask_ls]

            # Get viewing frustums for all cameras to only select points inside these [(frustum_edges, frustum_normals), (...)]
            frustum_ls = [
                get_frustum_torch(camera=cam, edge2camera=cam_ed, edge2reference=wo2veh, device="cpu")
                for
                (cam, cam_ed) in
                zip(cam_nodes, cam_ed_ls)
            ]

            # Reduce point cloud size by subsampling
            positive_only=False
            if scene.scene_descriptor.get('pt_cloud_fix', False):
                pt_cloud_li = self.reduce_point_cloud_size(pt_cloud_li, f, positive_return_only=positive_only, pc_frame_id=pc_frame_id, downsampling_factor=0.1, floor=0.2)
            else:
                pt_cloud_li = self.reduce_point_cloud_size(pt_cloud_li, f, positive_return_only=positive_only, pc_frame_id=pc_frame_id, downsampling_factor=0.1, floor=None)

            # Get a mask for points inside the viewing frustum
            point_frustum_mask_ls = []
            for i in range(len(frustum_ls)):
                frustum_normals = frustum_ls[i][1].detach().numpy()
                point_plane_dist = np.sum(
                    (pt_cloud_li - ray_o.clone().detach().numpy()[cam_frame_mask_ls[i]][None, 0])[None, ...] * frustum_normals[:, None], axis=-1)
                point_frustum_mask = np.where(np.all(point_plane_dist <= 0, axis=0))
                point_frustum_mask_ls.append(point_frustum_mask)

            output_dict = self.compute_dist_ray_point(pt_cloud_li,
                                                      point_masks=point_frustum_mask_ls,
                                                      ray_o=ray_o,
                                                      ray_d=ray_d,
                                                      ray_bundle=ray_bundle,
                                                      frame_mask=frame_mask,
                                                      cam_frame_mask_ls=cam_frame_mask_ls,
                                                      join_cams=False,
                                                      veh2world=veh2wo,
                                                      frame0_trafo=scene.frames[0].global_transformation,
                                                      frame_ls=fr_idx,)

            output_dict.update({'frame_mask': tuple(
                [torch.tensor(frame_mask[0], device='cpu', dtype=torch.int32), torch.tensor(frame_mask[1], device='cpu',
                                                                                            dtype=torch.int64)])})

            # Create the same outputs as previously
            for cf in c_fr_idx:
                if int(cf[1]) == int(f):
                    cf = tuple(cf.tolist())
                    pt_cloud_select[cf] = pt_cloud_li

            closest_point_mask_new = {}
            for cf in c_fr_idx:
                if int(cf[1]) == int(f):
                    cf_mask = torch.where(torch.all(ray_bundle.xys[frame_mask][..., [2, 3]] == cf[None], dim=-1))
                    cf = tuple(cf.tolist())
                    # To achieve a consistent Light Field parametrization across the full scene we  transform ray direction from the local vehicle coordinates back into the world coordinates
                    ray_dirs_cf_veh = output_dict['ray_d'][cf_mask]
                    veh2world = veh2wo[:3, :3]

                    ray_dirs_cf_world = torch.matmul(veh2world, ray_dirs_cf_veh.T).T
                    ray_dirs[cf] = ray_dirs_cf_world

                    closest_point_mask[cf] = output_dict['point_mask_closest'][cf_mask]

                    closest_point_dist[cf] = output_dict['distance'][cf_mask]
                    closest_point_azimuth[cf] = output_dict['azimuth'][cf_mask]
                    closest_point_pitch[cf] = output_dict['pitch'][cf_mask]

            pt_cloud_select = {k: torch.tensor(v, device='cpu', dtype=torch.float32) for k, v in
                               pt_cloud_select.items()}

        return ray_bundle, closest_point_mask, [pt_cloud_select, closest_point_dist, closest_point_azimuth,
                                                closest_point_pitch, output_dict], ray_dirs,

    def compute_dist_ray_point(self, points, point_masks, ray_o, ray_d, ray_bundle, cam_frame_mask_ls, frame_mask,
                               join_cams=True,
                               veh2world=np.eye(4), frame0_trafo=np.eye(4), frame_ls=[]):

        points = torch.tensor(points, dtype=ray_o.dtype, device=ray_o.device)

        if join_cams:
            # Could speed up computation and will reduce complexity for similar camera poses
            point_masks = [
                tuple(
                    np.unique(np.concatenate(
                        [mask[0] for mask in point_masks]
                    ))[None]
                )
            ]
            cam_frame_mask_ls = [frame_mask]

        output_dicts_ls = []
        # If computational necessary use chunks n_pts * n_rays
        for k, pt_mask in enumerate(point_masks):
            pts_masked = points[pt_mask]

            n_rays = ray_o[cam_frame_mask_ls[k]].shape[0]
            n_pts = len(pt_mask[0])
            if n_pts == 0:
                print(ray_bundle.xys[0, 0])
                print(frame_ls)
            rays_per_chunk = int(np.ceil(self._pt_chunk_sz / n_pts))
            n_chunks = int(np.ceil(n_rays / rays_per_chunk))
            if n_chunks > 1:
                print(n_chunks)
                print(self._pt_chunk_sz)

            for i in range(n_chunks):
                chunk_dict = {}
                chunk_mask_ray = tuple(
                    [v[i * rays_per_chunk: (i + 1) * rays_per_chunk] for v in cam_frame_mask_ls[k]])

                ray_o_chunk = ray_o[chunk_mask_ray]
                if torch.var(ray_o_chunk, axis=0).sum() < 1e-5:
                    all_origins = False
                else:
                    all_origins = True

                ray_d_chunk = ray_d[chunk_mask_ray]

                # pts_masked = torch.tensor(pts_masked, device=ray_o.device, dtype=ray_o.dtype)

                n_frustum_pts = len(pts_masked)
                n_rays = len(ray_d_chunk)
                if not all_origins:
                    # Vector between camera origin and all pts
                    cam_2_pts = pts_masked - ray_o_chunk[0]
                    direct_len_cam2pts = torch.norm(cam_2_pts, dim=1)
                    cam_pts_dir = cam_2_pts / direct_len_cam2pts[:, None]
                    # Cos of angle between ray and point2camorigin # From here on it blows up
                    cos_ray_pts = torch.matmul(ray_d_chunk, cam_pts_dir.T)
                else:
                    cam_2_pts = pts_masked[None] - ray_o_chunk[:, None]
                    direct_len_cam2pts = torch.norm(cam_2_pts, dim=-1)
                    cam_pts_dir = cam_2_pts / direct_len_cam2pts[..., None] # [n_rays, n_pts, 3]

                    # Cos of angle between ray and point2camorigin # From here on it blows up
                    cos_ray_pts = torch.cat(
                        [torch.matmul(ray_dir[None], pt_dir.T) for ray_dir, pt_dir in zip(ray_d_chunk, cam_pts_dir)], dim=0)

                eps = 1e-4
                # Prevent negative cosine
                cos_ray_pts[cos_ray_pts >= torch.Tensor([1.])] = 1. - eps

                # "Correct" equation
                #         # srt(1 - x**2) == sin(arccos(x))
                #         sin_pitch = np.sqrt(1 - cos_ray_pts ** 2)
                #         # Orthogonal distance from ray to each point
                #         ray2pts_dist = sin_pitch * direct_len_cam2pts[None]
                #         # Sort all points by distance from the ray
                #         mask_distance_sorted = np.argpartition(ray2pts_dist, self.k_closest)
                # Simplification square root and root as monotone function
                if not all_origins:
                    abstracted_distance = (1 - cos_ray_pts) * direct_len_cam2pts[None]
                else:
                    abstracted_distance = (1 - cos_ray_pts) * direct_len_cam2pts

                mask_distance_sorted = torch.topk(-abstracted_distance, self.k_closest)[1]


                closest_mask = tuple([torch.linspace(0,
                                                     ray_o_chunk.shape[0] - 1,
                                                     ray_o_chunk.shape[0],
                                                     dtype=torch.int64)[:, None].repeat(1, self.k_closest),
                                      mask_distance_sorted])

                closest_pts_mask = pt_mask[0][closest_mask[1].to(torch.int32)]

                # pt_ray_dist_sort = np.sort(pts_ray_dist)
                # pt_ray_dist_sort = pts_ray_dist[distance_sort_mask]
                # Get distance information for all point ray pairs
                closest_cos_ray_pts = cos_ray_pts[closest_mask]

                # pt_ray_dist_closest = ray2pts_dist[closest_mask]
                pitch_closest = torch.arccos(closest_cos_ray_pts)
                sin_pitch_closest = torch.sqrt(1 - closest_cos_ray_pts ** 2)
                if not all_origins:
                    pt_ray_dist_closest = sin_pitch_closest * direct_len_cam2pts[closest_mask[1]]
                else:
                    pt_ray_dist_closest = sin_pitch_closest * direct_len_cam2pts[closest_mask]

                # ray_length_closest = ray_length_closest[closest_mask]
                ray_length_closest = closest_cos_ray_pts * torch.norm((pts_masked - ray_o_chunk[0])[closest_mask[1]], dim=-1)

                points_k_closest = points[closest_pts_mask]

                # Normal axis of plane
                n = ray_d[chunk_mask_ray]

                if self.new_enc:
                    # Encode angle in world coordinates
                    points_veh = torch.cat([points_k_closest, torch.ones([n_rays, self.k_closest, 1])], dim=-1)

                    points_world = torch.matmul(veh2world, points_veh.reshape([-1, 4]).T)[:3].T.reshape(
                        n_rays, self.k_closest, 3)
                    # Center in frame0
                    points_world = points_world - torch.tensor(frame0_trafo, dtype=ray_o.dtype, device=ray_o.device)[:3, 3]

                    norm_pts = torch.norm(points_world, dim=-1)
                    points_world_normalized = points_world / norm_pts[..., None]

                    ray_chunk_wo = torch.matmul(veh2world[:3, :3], ray_d_chunk.T).T
                    closest_cos_ray_pts = (ray_chunk_wo[:, None] * points_world_normalized).sum(dim=-1)

                    eps = 1e-4
                    # Prevent negative cosine
                    closest_cos_ray_pts[closest_cos_ray_pts >= torch.tensor([1.])] = 1. - eps

                    pitch_closest = torch.arccos(closest_cos_ray_pts)

                # Vector from orthogonal projection of closest points on a ray to that point
                vec_ray2point = points_k_closest - (
                            ray_o_chunk[:, None] + n[..., None, :] * ray_length_closest[..., None])

                # Upward pointing lidar axis
                # TODO: Check with waymo
                y_axis = torch.tensor([[0, 1., 0.]], dtype=ray_o.dtype, device=ray_o.device)

                # Define plane normal to ray and projection on it
                e1 = y_axis - torch.sum(y_axis * n, dim=-1)[:, None] * n
                e1 = (1 / torch.norm(e1, dim=-1)[:, None]) * e1
                e2 = torch.cross(n, e1)

                projection_mat = torch.cat([e1[:, None], e2[:, None]], dim=1)

                # Project vector from ray to point in this plane and calculate azimuth angle
                ray2point_plane_projected = torch.matmul(projection_mat[:, None], vec_ray2point[..., None]).squeeze()
                ray2point_plane_projected = (1 / torch.norm(ray2point_plane_projected, dim=-1))[
                                                ..., None] * ray2point_plane_projected

                azimuth_closest = torch.atan2(ray2point_plane_projected[..., 1], ray2point_plane_projected[..., 0])

                # Resort ray_bundle to that chunk
                if not self.new_enc:
                    ray_bundle.lengths[chunk_mask_ray] = ray_length_closest

                # Store everything in chunk dict
                chunk_dict.update(
                    {
                        "distance": pt_ray_dist_closest,
                        "projected_distance": ray_length_closest,
                        "pitch": pitch_closest,
                        "azimuth": azimuth_closest,
                        "ray_d": ray_d_chunk,
                        "point_mask_closest": torch.tensor(closest_pts_mask, device=ray_o.device, dtype=torch.int64),
                        'mask_ray_0': torch.tensor(chunk_mask_ray[0], device=ray_o.device, dtype=torch.int64),
                        'mask_ray_1': torch.tensor(chunk_mask_ray[0], device=ray_o.device, dtype=torch.int64),
                    }
                )
                output_dicts_ls.append(chunk_dict)

        output_dict = {k: torch.cat([dict[k] for dict in output_dicts_ls], dim=0) for k in output_dicts_ls[0].keys()}

        return output_dict
        # TODO: Parallelize by adding batch per camera dimension, when not joining masks

    def reduce_point_cloud_size(self, points, frame_id, pc_frame_id,  positive_return_only=True, downsampling_factor=0.1, floor=None):
        # Downsample and reduce point cloud if requested

        if not (self._pt_caching and pc_frame_id in self._pts_dict):
            if positive_return_only:
                # Only return points in front of vehicle
                # TODO: Find different solution for WAYMO e.g. just the viewing frustum for Left and Right Camera
                points = points[np.where(points[:, 0] >= 0.)]

            if downsampling_factor is not None:
                if floor is None:
                    all_points = o3d.geometry.PointCloud()
                    all_points.points = o3d.utility.Vector3dVector(points)
                    # Reduce Point cloud size
                    all_points_sampled = all_points.voxel_down_sample(voxel_size=downsampling_factor)
                else:
                    all_points_ground = o3d.geometry.PointCloud()
                    all_points_ground.points = o3d.utility.Vector3dVector(points[points[..., 2] < 0.2])
                    all_points_ground = all_points_ground.voxel_down_sample(voxel_size=floor)
                    all_points_above = o3d.geometry.PointCloud()
                    all_points_above.points = o3d.utility.Vector3dVector(points[points[..., 2] > 0.2])
                    all_points_above = all_points_above.voxel_down_sample(voxel_size=downsampling_factor)

                    all_points_sampled = all_points_ground + all_points_above

                if len(all_points_sampled.points) > self.n_pts:
                    all_points = all_points_sampled
                else:
                    Warning(
                        'Down sampling factor {} is to big for frame {}.'.format(downsampling_factor, int(frame_id)))
            else:
                all_points = o3d.geometry.PointCloud()
                all_points.points = o3d.utility.Vector3dVector(points)

            all_points = all_points.uniform_down_sample(every_k_points=len(all_points.points) // self.n_pts)
            points = np.asarray(all_points.points)
            if self._pt_caching:
                self._pts_dict[pc_frame_id] = points
        elif pc_frame_id in self._pts_dict:
            points = self._pts_dict[pc_frame_id]

        # Reduce point cloud to be always the same size
        pts_mask_id = np.random.choice(np.linspace(0, len(points) - 1, len(points), dtype=int),
                                       self.n_pts,
                                       replace=False)
        points = points[pts_mask_id]

        return points


# TODO: If possible more efficient to achieve similar speed distributed on the CPU and with Numpy!
class PointDistanceRaysamplerNP(nn.Module):

    def __init__(self,
                 n_pts: int,
                 k_closest: int,
                 point_chunk_size: int = 1e12,
                 merge_pcd: bool = False,
                 n_frames_merged=5,
                 augment_point_cloud=False,
                 new_encoding=False,
                 pt_caching=False,
                 ):

        super(PointDistanceRaysamplerNP, self).__init__()

        self.n_pts = n_pts
        self.k_closest = k_closest
        self._pt_chunk_sz = point_chunk_size
        self.merge_pcd = merge_pcd
        if merge_pcd:
            self.icp = ICP(n_frames=n_frames_merged,)
            self.augment = augment_point_cloud
            self._pt_caching = pt_caching

        self.new_enc = new_encoding

    def forward(self,
                ray_bundle: RayBundle,
                scene,
                validation,):

        device = ray_bundle.origins.device

        # Prepare ray_bundle.lengths to keep the point projection along each ray
        if not self.new_enc:
            ray_bundle = ray_bundle._replace(
                lengths=torch.zeros([ray_bundle.lengths.shape[0], ray_bundle.lengths.shape[1], self.k_closest],
                                    device=device))
        xycfn = ray_bundle.xys.cpu().numpy()

        # Get camera and frame combinations
        c_fr_idx = ray_bundle.xys[..., [2, 3]].unique(dim=1)[0]
        fr_idx = ray_bundle.xys[..., 3].unique(dim=1)[0]

        lidar_idx = list(scene.nodes['lidar'])
        lidar = scene.nodes['lidar'][lidar_idx[0]]

        closest_point_mask = {}
        closest_point_dist = {}
        closest_point_azimuth = {}
        closest_point_pitch = {}
        pt_cloud_select = {}
        pt_cloud_select_new = {}
        ray_dirs = {}

        # Iterate over frames
        for f in fr_idx:
            f = int(f)
            try:
                frame = scene.frames[int(f)]
            except:
                KeyError('Frame with the ID {} does not exist for this scene'.format(int(f)))

            # Select all rays from the same frame
            if ray_bundle.xys.dim() >= 4:
                frame_mask = np.where(xycfn[..., 0, 3] == f)
            else:
                frame_mask = np.where(xycfn[..., 3] == f)

            # Relevant cameras and their edges in the scene graph of the current frame
            cam_idx = np.unique(xycfn[frame_mask][..., 2])
            cam_nodes = [scene.nodes['camera'][int(c)] for c in cam_idx]
            cam_ed_ls = [scene.frames[int(f)].get_edge_by_child_idx([int(c)])[0][0] for c in cam_idx]

            # Get ray origins and ray directions for these rays
            ray_o = ray_bundle.origins.detach().numpy()
            ray_d = ray_bundle.directions.detach().numpy()

            li_sel_idx = np.array([v.scene_idx if v.name == 'TOP' else -1 for k, v in scene.nodes['lidar'].items()])
            top_lidar_id = li_sel_idx[np.where(li_sel_idx > 0)][0]
            # TODO: Add point cloud loading and icp to get_item of the scene similar to image pass
            if not self.merge_pcd:
                # Use point cloud of this frame
                pt_cloud_li = scene.frames[int(f)].load_point_cloud(top_lidar_id)[..., :3]
                # pt_cloud_li = pt_cloud_li[np.where(pt_cloud_li[..., 0] >= 0.)]
                pc_frame_id = int(f)

            else:
                augment = self.augment and not validation
                # Use point cloud merged across frames
                pt_cloud_li, _, camera_trafo, pc_frame_id = self.icp(scene, cam_frame_id=int(f), caching=False, augment=augment)

                if augment:
                    # change camera according to the augmentad new point cloud frame
                    ray_o = camera_trafo.dot(np.concatenate([ray_o, np.ones_like(ray_o)[..., :1]], axis=-1).squeeze().T).T[None, ..., :3]
                    ray_d = camera_trafo[:3, :3].dot(ray_d.squeeze().T).T[None]

            # Get camera poses in local point cloud corrdinates and the camera viewing frustum
            veh2wo = scene.frames[pc_frame_id].global_transformation
            wo2veh = np.concatenate(
                [veh2wo[:3, :3].T,
                 veh2wo[:3, :3].T.dot(-veh2wo[:3, 3])[:, None]],
                axis=1
            )

            wo2veh = np.concatenate([wo2veh, np.array([[0., 0., 0., 1.]])])


            cam_mask_ls = [np.where(xycfn[frame_mask][..., 2] == int(c)) for c in cam_idx]
            cam_frame_mask_ls = [tuple([v[cam_mask] for v in frame_mask]) for cam_mask in cam_mask_ls]

            # Get viewing frustums for all cameras to only select points inside these [(frustum_edges, frustum_normals), (...)]
            frustum_ls = [
                get_frustum(camera=cam, edge2camera=cam_ed, edge2reference=wo2veh, device="cpu")
                for
                (cam, cam_ed) in
                zip(cam_nodes, cam_ed_ls)
            ]

            # Reduce point cloud size by subsampling
            if scene.scene_descriptor.get('pt_cloud_fix', False):
                pt_cloud_li = self.reduce_point_cloud_size(pt_cloud_li, f, downsampling_factor=0.1, floor=0.2)
            else:
                pt_cloud_li = self.reduce_point_cloud_size(pt_cloud_li, f, downsampling_factor=0.1, floor=None)

            # Get a mask for points inside the viewing frustum
            point_frustum_mask_ls = []
            for i in range(len(frustum_ls)):
                frustum_normals = frustum_ls[i][1].numpy()
                point_plane_dist = np.sum(
                    (pt_cloud_li - ray_o[cam_frame_mask_ls[i]][None, 0])[None, ...] * frustum_normals[:, None], axis=-1)
                point_frustum_mask = np.where(np.all(point_plane_dist <= 0, axis=0))
                point_frustum_mask_ls.append(point_frustum_mask)

            output_dict = self.compute_dist_ray_point(pt_cloud_li,
                                                      point_masks=point_frustum_mask_ls,
                                                      ray_o=ray_o,
                                                      ray_d=ray_d,
                                                      ray_bundle=ray_bundle,
                                                      frame_mask=frame_mask,
                                                      cam_frame_mask_ls=cam_frame_mask_ls,
                                                      join_cams=False,
                                                      veh2world=veh2wo,
                                                      frame0_trafo=scene.frames[0].global_transformation)

            output_dict.update({'frame_mask': tuple(
                [torch.tensor(frame_mask[0], device='cpu', dtype=torch.int32), torch.tensor(frame_mask[1], device='cpu',
                                                                                            dtype=torch.int64)])})

            # Create the same outputs as previously
            for cf in c_fr_idx:
                if int(cf[1]) == int(f):
                    cf = tuple(cf.tolist())
                    pt_cloud_select[cf] = pt_cloud_li


            closest_point_mask_new = {}
            for cf in c_fr_idx:
                if int(cf[1]) == int(f):
                    cf_mask = torch.where(torch.all(ray_bundle.xys[frame_mask][..., [2, 3]] == cf[None], dim=-1))
                    cf = tuple(cf.tolist())
                    # To achieve a consistent Light Field parametrization across the full scene we  transform ray direction from the local vehicle coordinates back into the world coordinates
                    ray_dirs_cf_veh = output_dict['ray_d'][cf_mask]
                    veh2world = torch.tensor(veh2wo[:3, :3], dtype=ray_dirs_cf_veh.dtype)

                    ray_dirs_cf_world = torch.matmul(veh2world, ray_dirs_cf_veh.T).T
                    ray_dirs[cf] = ray_dirs_cf_world

                    closest_point_mask[cf] = output_dict['point_mask_closest'][cf_mask]

                    closest_point_dist[cf] = output_dict['distance'][cf_mask]
                    closest_point_azimuth[cf] = output_dict['azimuth'][cf_mask]
                    closest_point_pitch[cf] = output_dict['pitch'][cf_mask]

            pt_cloud_select = {k: torch.tensor(v, device='cpu', dtype=torch.float32) for k, v in pt_cloud_select.items()}

        return ray_bundle, closest_point_mask, [pt_cloud_select, closest_point_dist, closest_point_azimuth,
                                                closest_point_pitch, output_dict], ray_dirs,

    def compute_dist_ray_point(self, points, point_masks, ray_o, ray_d, ray_bundle, cam_frame_mask_ls, frame_mask, join_cams=True,
                               veh2world=np.eye(4), frame0_trafo=np.eye(4)):
        if join_cams:
            # Could speed up computation and will reduce complexity for similar camera poses
            point_masks = [
                tuple(
                    np.unique(np.concatenate(
                        [mask[0] for mask in point_masks]
                    ))[None]
                )
            ]
            cam_frame_mask_ls = [frame_mask]

        output_dicts_ls = []
        # If computational necessary use chunks n_pts * n_rays
        for k, pt_mask in enumerate(point_masks):
            pts_masked = points[pt_mask]

            n_rays = ray_o[cam_frame_mask_ls[k]].shape[0]
            n_pts = len(pt_mask[0])
            if n_pts == 0:
                print(ray_bundle.xys[0,0])
            rays_per_chunk = int(np.ceil(self._pt_chunk_sz / n_pts))
            n_chunks = int(np.ceil(n_rays / rays_per_chunk))
            if n_chunks > 1:
                print(n_chunks)
                print(self._pt_chunk_sz)

            for i in range(n_chunks):
                chunk_dict = {}
                chunk_mask_ray = tuple(
                    [v[i * rays_per_chunk: (i + 1) * rays_per_chunk] for v in cam_frame_mask_ls[k]])

                ray_o_chunk = ray_o[chunk_mask_ray]
                assert np.var(ray_o_chunk, axis=0).sum() < 1e-5

                ray_d_chunk = ray_d[chunk_mask_ray]

                pts_masked = pts_masked

                n_frustum_pts = len(pts_masked)
                n_rays = len(ray_d_chunk)
                # Vector between camera origin and all pts
                cam_2_pts = pts_masked - ray_o_chunk[0]
                direct_len_cam2pts = np.linalg.norm(cam_2_pts, axis=1)
                cam_pts_dir = cam_2_pts / direct_len_cam2pts[:, None]

                # Cos of angle between ray and point2camorigin # From here on it blows up
                cos_ray_pts = ray_d_chunk.dot(cam_pts_dir.T)

                eps = 1e-4
                # Prevent negative cosine
                cos_ray_pts[cos_ray_pts >= np.array([1.])] = 1. - eps

                # "Correct" implementation
                #         # srt(1 - x**2) == sin(arccos(x))
                #         sin_pitch = np.sqrt(1 - cos_ray_pts ** 2)
                #         # Orthogonal distance from ray to each point
                #         ray2pts_dist = sin_pitch * direct_len_cam2pts[None]
                #         # Sort all points by distance from the ray
                #         mask_distance_sorted = np.argpartition(ray2pts_dist, self.k_closest)
                # Simplification square root and root as monotone function
                abstracted_distance = (1 - cos_ray_pts) * direct_len_cam2pts[None]
                mask_distance_sorted = np.argpartition(abstracted_distance, self.k_closest)

                distance_sort_mask = tuple(
                    [
                        np.repeat(
                            np.linspace(0, ray_o_chunk.shape[0] - 1, ray_o_chunk.shape[0], dtype=int)[:, None],
                              cam_pts_dir.shape[0], axis=1
                        ),
                        mask_distance_sorted
                    ])

                closest_mask = tuple([distance_sort_mask[0][:, :self.k_closest],
                                     distance_sort_mask[1][:, :self.k_closest]])

                closest_pts_mask = pt_mask[0][closest_mask[1]]

                # pt_ray_dist_sort = np.sort(pts_ray_dist)
                # pt_ray_dist_sort = pts_ray_dist[distance_sort_mask]
                # Get distance information for all point ray pairs
                closest_cos_ray_pts = cos_ray_pts[closest_mask]

                # pt_ray_dist_closest = ray2pts_dist[closest_mask]
                pitch_closest = np.arccos(closest_cos_ray_pts)
                sin_pitch_closest = np.sqrt(1 - closest_cos_ray_pts ** 2)
                pt_ray_dist_closest = sin_pitch_closest * direct_len_cam2pts[closest_mask[1]]

                # ray_length_closest = ray_length_closest[closest_mask]
                ray_length_closest = closest_cos_ray_pts * np.linalg.norm(cam_2_pts[closest_mask[1]], axis=-1)

                points_k_closest = points[closest_pts_mask]

                # Normal axis of plane
                n = ray_d[chunk_mask_ray]

                if self.new_enc:
                    # Encode angle in world coordinates
                    points_veh = np.concatenate([points_k_closest, np.ones([n_rays, self.k_closest, 1])], axis=-1)

                    points_world = np.matmul(veh2world, points_veh.reshape([-1, 4]).T)[:3].T.reshape(
                        n_rays, self.k_closest, 3)
                    # Center in frame0
                    points_world = points_world - frame0_trafo[:3, 3]

                    norm_pts = np.linalg.norm(points_world, axis=-1)
                    points_world_normalized = points_world / norm_pts[..., None]

                    ray_chunk_wo = np.matmul(veh2world[:3, :3], ray_d_chunk.T).T
                    closest_cos_ray_pts = (ray_chunk_wo[:, None] * points_world_normalized).sum(axis=-1)

                    eps = 1e-4
                    # Prevent negative cosine
                    closest_cos_ray_pts[closest_cos_ray_pts >= np.array([1.])] = 1. - eps

                    pitch_closest = np.arccos(closest_cos_ray_pts)

                # Vector from orthogonal projection of closest points on a ray to that point
                vec_ray2point = points_k_closest - (ray_o_chunk[:, None] + n[..., None, :] * ray_length_closest[..., None])

                # Upward pointing lidar axis
                # TODO: Check with waymo
                y_axis = np.array([[0, 1., 0.]])

                # Define plane normal to ray and projection on it
                e1 = y_axis - np.sum(y_axis * n, axis=-1)[:, None] * n
                e1 = (1 / np.linalg.norm(e1, axis=-1)[:, None]) * e1
                e2 = np.cross(n, e1)

                projection_mat = np.concatenate([e1[:, None], e2[:, None]], axis=1)

                # Project vector from ray to point in this plane and calculate azimuth angle
                ray2point_plane_projected = np.matmul(projection_mat[:, None], vec_ray2point[..., None]).squeeze()
                ray2point_plane_projected = (1 / np.linalg.norm(ray2point_plane_projected, axis=-1))[
                                          ..., None] * ray2point_plane_projected

                azimuth_closest = np.arctan2(ray2point_plane_projected[..., 1], ray2point_plane_projected[..., 0])

                # Resort ray_bundle to that chunk
                if not self.new_enc:
                    ray_bundle.lengths[chunk_mask_ray] = torch.tensor(ray_length_closest,
                                                                      dtype=ray_bundle.lengths.dtype, device='cpu')
                # Store everything in chunk dict
                chunk_dict.update(
                    {
                        "distance": pt_ray_dist_closest,
                        "projected_distance": ray_length_closest,
                        "pitch": pitch_closest,
                        "azimuth": azimuth_closest,
                        "ray_d": ray_d_chunk,
                        "point_mask_closest": closest_pts_mask,
                        'mask_ray_0': chunk_mask_ray[0],
                        'mask_ray_1': chunk_mask_ray[1],
                    }
                )
                output_dicts_ls.append(chunk_dict)

        output_dict = {k: torch.tensor(np.concatenate([dict[k] for dict in output_dicts_ls], axis=0), device='cpu',
                                       dtype=torch.float32 if (output_dicts_ls[0][k].dtype == 'float64' or
                                                              output_dicts_ls[0][k].dtype == 'float32')
                                       else torch.int64) for k in output_dicts_ls[0].keys()}

        return output_dict
        # TODO: Parallelize by adding batch per camera dimension, when not joining masks

    def reduce_point_cloud_size(self, points, frame_id, positive_return_only=True, downsampling_factor=0.1, floor=None):
        # Downsample and reduce point cloud if requested

        if positive_return_only:
            # Only return points in front of vehicle
            # TODO: Find different solution for WAYMO e.g. just the viewing frustum for Left and Right Camera
            points = points[np.where(points[:, 0] >= 0.)]

        if downsampling_factor is not None:
            if floor is None:
                all_points = o3d.geometry.PointCloud()
                all_points.points = o3d.utility.Vector3dVector(points)
                # Reduce Point cloud size
                all_points_sampled = all_points.voxel_down_sample(voxel_size=downsampling_factor)
            else:
                all_points_ground = o3d.geometry.PointCloud()
                all_points_ground.points = o3d.utility.Vector3dVector(points[points[..., 2] < 0.2])
                all_points_ground = all_points_ground.voxel_down_sample(voxel_size=floor)
                all_points_above = o3d.geometry.PointCloud()
                all_points_above.points = o3d.utility.Vector3dVector(points[points[..., 2] > 0.2])
                all_points_above = all_points_above.voxel_down_sample(voxel_size=downsampling_factor)

                all_points_sampled = all_points_ground + all_points_above

            if len(all_points_sampled.points) > self.n_pts:
                all_points = all_points_sampled
            else:
                Warning(
                    'Down sampling factor {} is to big for frame {}.'.format(downsampling_factor, int(frame_id)))
        else:
            all_points = o3d.geometry.PointCloud()
            all_points.points = o3d.utility.Vector3dVector(points)

        all_points = all_points.uniform_down_sample(every_k_points=len(all_points.points) // self.n_pts)
        points = np.asarray(all_points.points)

        # Reduce point cloud to be always the same size
        pts_mask_id = np.random.choice(np.linspace(0, len(points) - 1, len(points), dtype=int),
                                       self.n_pts,
                                       replace=False)
        points = points[pts_mask_id]

        return points

    def plot_point_cloud_3d(self, ray_o, ray_d, points, point_masks, n_sel=50000, pt_mask_sort=None,
                            selected_rays=True):

        ray_id = 0
        if pt_mask_sort is not None:
            close_points = points[pt_mask_sort[:, :self.k_closest]]
            if selected_rays:
                ray_id = torch.randint(0, len(close_points), (128,))
                close_points = close_points[ray_id]
            close_points = close_points.view(-1, 3).detach().cpu().numpy()

        if selected_rays:
            cam_ray_plt = ray_o[0, :, None] + ray_d[0, :, None] * torch.linspace(0, 75, 20, device=ray_o.device)[None,
                                                                  :, None]
            cam_ray_plt = cam_ray_plt[ray_id]
        else:
            cam_ray_plt = ray_o[0, :, None] + ray_d[0, :, None] * torch.linspace(0, 75, 4, device=ray_o.device)[None,
                                                                  :, None]
        cam_ray_plt = cam_ray_plt.view(-1, 3).detach().cpu().numpy()

        pt_cloud_li_sel = points[torch.linspace(0, len(points), n_sel, dtype=torch.int64)]
        li_pt_plt = pt_cloud_li_sel[torch.where(torch.all(pt_cloud_li_sel[:, [0]].ge(-1.), dim=1))]
        li_pt_plt = li_pt_plt[torch.randint(0, len(li_pt_plt) - 1, (10000,))]
        li_pt_plt = li_pt_plt.detach().cpu().numpy()

        joined_masks = tuple(torch.cat([point_masks[0][0], point_masks[1][0]]).unique()[None])

        li_pt_frustum_plt = points[joined_masks].detach().cpu().numpy()

        fig3d = plt.figure()
        ax3d = fig3d.gca(projection='3d')
        if pt_mask_sort is not None:
            ax3d.scatter(close_points[:, 0], close_points[:, 1], close_points[:, 2], c='orange')
        ax3d.scatter(cam_ray_plt[:, 0], cam_ray_plt[:, 1], cam_ray_plt[:, 2], c='green')
        if not selected_rays:
            ax3d.scatter(li_pt_plt[:, 0], li_pt_plt[:, 1], li_pt_plt[:, 2], c='blue')
            ax3d.scatter(li_pt_frustum_plt[:, 0], li_pt_frustum_plt[:, 1], li_pt_frustum_plt[:, 2], c='magenta')


class BackgroundRaysampler(nn.Module):
    def __init__(
            self,
            camera_poses,
            n_pts_background: int,
            background_nodes,
    ):
        super().__init__()
        self._global_trafo = {}
        near_planes = {}
        far_planes = {}

        self._n_samples = n_pts_background

        transient_background = False
        for node in background_nodes.values():
            self._global_trafo[node.scene_idx] = node.transformation
            near_planes[node.scene_idx] = node.near
            far_planes[node.scene_idx] = node.far

        self._ray_bckg_sampler = RayPlaneIntersection(
            n_planes=self._n_samples,
            near=near_planes,
            far=far_planes,
            chunk_size=1e5,
            camera_poses=camera_poses,
            background_trafos=self._global_trafo,
            transient_background=transient_background,
        )

    def forward(
            self,
            ray_bundle: RayBundle,
            scene,
            intersections: torch.Tensor = None,
            obj_only: bool = False,
    ):
        bckg_node_id = 0

        lengths = self._ray_bckg_sampler(ray_bundle, scene, intersections, obj_only)
        lengths = torch.cat([ray_bundle.lengths, lengths], dim=-1)

        if ray_bundle.xys.dim() == 3:
            xycfn = ray_bundle.xys[..., None, :]
        else:
            xycfn = ray_bundle.xys

        xycfn_sh = list(xycfn.shape)
        xycfn_sh[-2] = self._n_samples

        if ray_bundle.xys.shape[-1] == 4:
            xycfn = xycfn.expand(xycfn_sh)
            new_nodes = torch.full(
                xycfn_sh[:-1], bckg_node_id, device=xycfn.device, dtype=xycfn.dtype
            )[..., None]
            xycfn = torch.cat([xycfn, new_nodes], dim=-1)
        else:
            new_nodes = xycfn[..., 0, :][..., None, :].expand(xycfn_sh)
            if xycfn.shape[-2] > 1:
                xycfn = torch.cat([xycfn, new_nodes], dim=-2)
            else:
                xycfn = new_nodes

        ray_bundle = ray_bundle._replace(lengths=lengths, xys=xycfn)

        return ray_bundle


class ObjectRaysampler(nn.Module):
    def __init__(
            self,
            class_nodes: dict,
            n_pts_object: int,
            sampling: str = None,
    ):
        super().__init__()

        self._n_samples = n_pts_object

        self._ray_obj_intersection = RayBoxIntersection(
            box_nodes=class_nodes,
            chunk_size=1e6,
        )
        self._obj_sampler = BoxRaysampler(
            n_pts_object,
            sampling=sampling,
        )

    def forward(self, ray_bundle: RayBundle, scene, **kwargs):
        """

        :param ray_bundle:
        :type ray_bundle:
        :param scene:
        :type scene:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        n_obj_nodes = None

        old_lengths_sh = ray_bundle.lengths.shape
        old_xycfn_sh = ray_bundle.xys.shape

        # Loop over all frames' scene graphs and convert ray in each dynamic leaf nodes coordinate frame
        (
            trafos_all,
            rot_all,
            scale_all,
            class_obj_idx,
            frame_mask_all,
        ) = self._get_trafos_for_all_intersect_candidates(scene, ray_bundle)

        # Split by frame
        frame_splits = [len(k) for k in class_obj_idx]
        trafos_w2o = trafos_all.get_matrix().split(frame_splits)
        rots_w2o = rot_all.get_matrix().split(frame_splits)
        scales_w2o = scale_all.get_matrix().split(frame_splits)

        (
            all_origins_o,
            all_directions_o,
            origins_o,
            directions_o,
        ) = self._transform_ray_bundle_to_object_node(
            ray_bundle, trafos_w2o, rots_w2o, scales_w2o, frame_mask_all, n_obj_nodes
        )
        n_possible_inter = len(all_origins_o) * self._n_samples
        node_intersections, ray_node_inter_idx = self._ray_obj_intersection(
            all_origins_o, all_directions_o, output_points=False
        )

        # Use Intersections to sample points inside objects bounding shape,
        # and extract a intersection mask sorted along rays length
        if not node_intersections[0] is None:
            if len(old_xycfn_sh) > 3:
                n_intersects_old = old_xycfn_sh[-2]
                global_samples = True
            else:
                n_intersects_old = 0
                global_samples = False

            # Sample points between intersections
            obj_z_vals_new = self._obj_sampler(
                box_in=node_intersections[0], box_out=node_intersections[1]
            )

            # Transform obj_z_vals back to world coordinates
            sampled_pts_o = (
                    all_origins_o[ray_node_inter_idx][:, None]
                    + all_directions_o[ray_node_inter_idx][:, None]
                    * obj_z_vals_new[..., None]
            )
            sampled_dirs_o = all_directions_o[ray_node_inter_idx][:, None, :].expand(
                sampled_pts_o.shape
            )

            # Create a mask to get intersecting rays and points from the full ray_bundle
            # Rearrange ray_node_inter_idx to ray_bundle sorting instead of frame sorted
            intersection_mask = self._get_ray_bundle_intersection_mask(
                ray_node_inter_idx, frame_mask_all, n_intersects_old
            )

            # Transform intersection points back to world coordinates
            (
                obj_z_vals_world,
                intersected_node_idx,
                sampling_pts_w,
            ) = self._transform_sampling_points_to_world(
                ray_bundle,
                sampled_pts_o,
                ray_node_inter_idx,
                trafos_w2o,
                rots_w2o,
                scales_w2o,
                frame_mask_all,
                origins_o,
                class_obj_idx,
            )

            lengths_new = ray_bundle.lengths
            # Additional empty slots for new intersections along each ray
            if old_lengths_sh[-1] + n_possible_inter > lengths_new.shape[-1]:
                add_inters = (
                        old_lengths_sh[-1] + n_possible_inter - ray_bundle.lengths.shape[-1]
                )

                lengths_new = torch.cat(
                    [
                        lengths_new,
                        torch.zeros(
                            list(old_lengths_sh[:-1]) + [add_inters],
                            device=lengths_new.device,
                        ),
                    ],
                    dim=-1,
                )

            lengths_new[intersection_mask[:-1]] = obj_z_vals_world

            xycfn_new = ray_bundle.xys
            # Additional empty slots for new intersections along each ray
            if not global_samples:
                xycfn_new = xycfn_new[..., None, :].clone().detach()

                add_inters = 1 + n_possible_inter - xycfn_new.shape[-2]
                new_nodes = ray_bundle.xys[..., None, :].clone().detach()
                new_nodes[..., -1] = -1
                xycfn_new = torch.cat(
                    [xycfn_new, new_nodes.repeat(1, 1, add_inters, 1)], dim=-2
                )
                xycfn_new = xycfn_new[..., 1:, :]

            elif n_intersects_old + n_possible_inter > xycfn_new.shape[-2]:
                add_inters = n_intersects_old + n_possible_inter - xycfn_new.shape[-2]
                new_nodes = ray_bundle.xys[..., 0, None, :].clone().detach()
                new_nodes[..., -1] = -1
                xycfn_new = torch.cat(
                    [xycfn_new, new_nodes.repeat(1, 1, add_inters, 1)], dim=-2
                )

            xycfn_new[intersection_mask] = intersected_node_idx.to(
                device=xycfn_new.device, dtype=xycfn_new.dtype
            )

            # Combine and sort all intersections
            ray_bundle, sorted_intersection_mask = self._sort_ray_bundle(
                ray_bundle, intersection_mask, lengths_new, xycfn_new
            )

            sampled_pts_o = sampled_pts_o.flatten(0, -2)
            sampled_dirs_o = sampled_dirs_o.flatten(0, -2)

            return ray_bundle, sorted_intersection_mask, sampled_pts_o, sampled_dirs_o
        else:
            return ray_bundle, None, None, None

    def _precache_mask(self, ray_bundle: RayBundle, scene, **kwargs):
        old_xycfn_sh = ray_bundle.xys.shape

        # Reduced version that just outputs a mask to get the scene intersecting rays from a respective ray_bundle
        ray_bundle = RayBundle(*[v.detach() for v in ray_bundle])

        # Loop over all frames' scene graphs and convert ray in each dynamic leaf nodes coordinate frame
        (
            trafos_all,
            rot_all,
            scale_all,
            class_obj_idx,
            frame_mask_all,
        ) = self._get_trafos_for_all_intersect_candidates(scene, ray_bundle)

        # Split by frame
        frame_splits = [len(k) for k in class_obj_idx]
        trafos_w2o = trafos_all.get_matrix().split(frame_splits)
        rots_w2o = rot_all.get_matrix().split(frame_splits)
        scales_w2o = scale_all.get_matrix().split(frame_splits)

        (
            all_origins_o,
            all_directions_o,
            _,
            _,
        ) = self._transform_ray_bundle_to_object_node(
            ray_bundle,
            trafos_w2o,
            rots_w2o,
            scales_w2o,
            frame_mask_all,
            n_obj_nodes=None,
        )

        node_intersections, ray_node_inter_idx = self._ray_obj_intersection(
            all_origins_o.detach(), all_directions_o.detach(), output_points=False
        )

        # Use Intersections to sample points inside objects bounding shape,
        # and extract a intersection mask sorted along rays length
        if not node_intersections[0] is None:
            if len(old_xycfn_sh) > 3:
                n_intersects_old = old_xycfn_sh[-2]
            else:
                n_intersects_old = 0

            # Create a mask to get intersecting rays and points from the full ray_bundle
            # Rearrange ray_node_inter_idx to ray_bundle sorting instead of frame sorted
            intersection_mask = self._get_ray_bundle_intersection_mask(
                ray_node_inter_idx, frame_mask_all, n_intersects_old
            )

            # Only extract elements relevant on ray_bundle level
            intersection_mask = [v[:, 0] for v in intersection_mask[:2]]

            return intersection_mask
        else:
            return None

    def _get_trafos_for_all_intersect_candidates(self, scene, ray_bundle):

        class_nodes = list(scene.nodes["object_class"].keys())
        object_nodes = list(scene.nodes["scene_object"].keys())

        frames_idx = ray_bundle.xys[..., 3].unique()

        new_trafo_list = []
        new_scale_list = []
        frame_mask_all = []
        class_obj_idx = []

        for f in frames_idx:
            try:
                frame = scene.frames[int(f)]
            except:
                KeyError(
                    "Frame with the ID {} does not exist for this scene".format(int(f))
                )

            # Select all rays from the same frame and calculate intersections with all objects in that frame
            if ray_bundle.xys.dim() >= 4:
                frame_mask = torch.where(ray_bundle.xys[..., 0, 3] == int(f))
            else:
                frame_mask = torch.where(ray_bundle.xys[..., 3] == int(f))

            frame_mask_all.append(frame_mask)
            scene_obj2obj_class = frame.get_edge_by_child_idx(class_nodes)

            frame_class_obj_idx = []
            # Get the transformations to each leaf node
            for c, edges2class in zip(class_nodes, scene_obj2obj_class):
                for edge in edges2class:
                    obj_j = edge.parent
                    if obj_j in object_nodes:
                        root2scene_obj = frame.get_edge_by_child_idx([obj_j])[0][0]
                        new_trafo_list.append(tuple([int(f), root2scene_obj.index]))
                        new_scale_list.append(tuple([int(f), edge.index]))
                        frame_class_obj_idx.append(torch.tensor([int(f), c, obj_j]))

            class_obj_idx.append(torch.stack(frame_class_obj_idx))

        trafo = scene.get_all_edge_transformations(new_trafo_list)
        rot = Rotate(trafo.get_matrix()[:, :3, :3], device=trafo.device)
        scale = scene.get_all_edge_scalings(new_scale_list)

        return trafo, rot, scale, class_obj_idx, frame_mask_all

    def _transform_ray_bundle_to_object_node(
            self,
            ray_bundle,
            trafos_w2o,
            rots_w2o,
            scales_w2o,
            frame_mask_all,
            n_obj_nodes=None,
    ):
        """

        :param ray_bundle:
        :type ray_bundle:
        :param trafos_w2o:
        :type trafos_w2o:
        :param rots_w2o:
        :type rots_w2o:
        :param scales_w2o:
        :type scales_w2o:
        :param frame_mask_all:
        :type frame_mask_all:
        :param n_obj_nodes:
        :type n_obj_nodes:
        :return:
        :rtype:
        """
        device = trafos_w2o[0].device

        if n_obj_nodes is None:
            n_obj_nodes = max([len(fr_trafo) for fr_trafo in trafos_w2o])

        origins_o = []
        directions_o = []

        # Transform Ray Bundle in each object nodes local frame
        for trafo, rot, scale, fr_mask in zip(
                trafos_w2o, rots_w2o, scales_w2o, frame_mask_all
        ):

            pt_w2obj = Transform3d(matrix=trafo).compose(Transform3d(matrix=scale))
            dir_w2obj = Transform3d(matrix=rot).compose(Transform3d(matrix=scale))

            frame_origins_o = pt_w2obj.transform_points(
                ray_bundle.origins[fr_mask].flatten(0, -2).to(device)
            )
            frame_directions_o = dir_w2obj.transform_points(
                ray_bundle.directions[fr_mask].flatten(0, -2).to(device)
            )

            # Normalize directions
            frame_directions_o = (
                    1
                    / torch.norm(frame_directions_o, dim=-1)[..., None]
                    * frame_directions_o
            )

            if frame_origins_o.ndim == 2:
                frame_origins_o = frame_origins_o[None]
                frame_directions_o = frame_directions_o[None]

            if len(frame_origins_o) < n_obj_nodes:
                # Pad with outside pointing arrays on a box corner to not introduce further intersections
                frame_origins_o = torch.cat(
                    [
                        frame_origins_o,
                        torch.full(
                            [
                                n_obj_nodes - frame_origins_o.shape[0],
                                frame_origins_o.shape[1],
                                3,
                            ],
                            1.0 + 1e-6,
                            device=frame_origins_o.device,
                        ),
                    ]
                )

                frame_directions_o = torch.cat(
                    [
                        frame_directions_o,
                        torch.full(
                            [
                                n_obj_nodes - frame_directions_o.shape[0],
                                frame_directions_o.shape[1],
                                3,
                            ],
                            1.0 + 1e-6,
                            device=frame_directions_o.device,
                        ),
                    ]
                )

            origins_o.append(frame_origins_o)
            directions_o.append(frame_directions_o)

        # Origins inobject coordinates and sorted by frames
        # From here on all tensors are sorted by frames and not like ray_bundle
        all_origins_o = torch.cat(origins_o, dim=1)
        all_directions_o = torch.cat(directions_o, dim=1)

        return all_origins_o, all_directions_o, origins_o, directions_o

    def _get_ray_bundle_intersection_mask(
            self, ray_node_inter_idx, frame_mask_all, n_intersects_old
    ):
        unordered_ray_inter_idx = torch.cat([fr_mask[1] for fr_mask in frame_mask_all])[
            ray_node_inter_idx[1]
        ]

        selected_ray_idx_new = unordered_ray_inter_idx[:, None].repeat(
            1, self._n_samples
        )
        intersection_pt_idx_new = (
                                          ray_node_inter_idx[0] * self._n_samples + n_intersects_old
                                  )[:, None] + torch.linspace(
            0,
            self._n_samples - 1,
            self._n_samples,
            device=ray_node_inter_idx[0].device,
            dtype=torch.int32,
        )

        intersection_mask = tuple(
            [
                torch.full(selected_ray_idx_new.shape, 0),
                selected_ray_idx_new,
                intersection_pt_idx_new,
                torch.full(selected_ray_idx_new.shape, 4),
            ]
        )

        return intersection_mask

    def _transform_sampling_points_to_world(
            self,
            ray_bundle,
            sampled_pts_o,
            ray_node_inter_idx,
            trafos_w2o,
            rots_w2o,
            scales_w2o,
            frame_mask_all,
            origins_o,
            class_obj_idx,
    ):
        """

        :param sampled_pts_o:
        :type sampled_pts_o:
        :param ray_bundle:
        :type ray_bundle:
        :param trafos_w2o:
        :type trafos_w2o:
        :param rots_w2o:
        :type rots_w2o:
        :param scales_w2o:
        :type scales_w2o:
        :return:
        :rtype:
        """

        node_inter_idx = ray_node_inter_idx[0]
        frame_ordered_ray_inter_idx = ray_node_inter_idx[1]
        unordered_ray_inter_idx = torch.cat([fr_mask[1] for fr_mask in frame_mask_all])[
            frame_ordered_ray_inter_idx
        ]

        sampling_pts_w = torch.zeros_like(sampled_pts_o, device=sampled_pts_o.device)
        intersected_node_idx = torch.full(sampled_pts_o.shape[:-1], -1)

        ordered_start_ray = 0
        for i, (trafo, rot, scale) in enumerate(zip(trafos_w2o, rots_w2o, scales_w2o)):
            ordered_end_ray = origins_o[i].shape[1] + ordered_start_ray

            # Just select intersections from rays in this frame
            ordered_fr_bool_mask = ray_node_inter_idx[1].ge(
                ordered_start_ray
            ) & ray_node_inter_idx[1].le(ordered_end_ray - 1)
            ordered_fr_mask = tuple(
                [
                    node_inter_idx[ordered_fr_bool_mask],
                    frame_ordered_ray_inter_idx[ordered_fr_bool_mask]
                    - ordered_start_ray,
                ]
            )

            # Transform intersection pts back to world space
            pt_objw = (
                Transform3d(matrix=trafo).compose(Transform3d(matrix=scale)).inverse()
            )
            frame_pts_w = torch.zeros(
                [len(trafo), ordered_end_ray - ordered_start_ray, self._n_samples, 3],
                device=sampled_pts_o.device,
            )
            frame_pts_w[ordered_fr_mask] = sampled_pts_o[ordered_fr_bool_mask]
            frame_pts_w = pt_objw.transform_points(
                frame_pts_w.view(len(trafo), -1, 3)
            ).view(len(trafo), -1, self._n_samples, 3)

            sampling_pts_w[ordered_fr_bool_mask] = frame_pts_w[ordered_fr_mask]

            # Strore intersected rays for each point
            intersected_node_idx[ordered_fr_bool_mask] = class_obj_idx[i][
                                                             ordered_fr_mask[0], 2
                                                         ][:, None].repeat(1, self._n_samples)

            # Set starting point for next frame
            ordered_start_ray = ordered_end_ray

        obj_z_vals_world = (
                sampling_pts_w - ray_bundle.origins[0, unordered_ray_inter_idx][:, None]
        )

        obj_z_vals_world = torch.norm(obj_z_vals_world, dim=-1)

        return obj_z_vals_world, intersected_node_idx, sampling_pts_w

    def _sort_ray_bundle(
            self, ray_bundle, intersection_mask, lengths_new=None, xycfn_new=None
    ):
        """

        :param ray_bundle:
        :type ray_bundle:
        :param intersection_mask:
        :type intersection_mask:
        :param lengths_new:
        :type lengths_new:
        :return:
        :rtype:
        """

        if lengths_new is None:
            lengths = ray_bundle.lengths
        else:
            lengths = lengths_new

        if xycfn_new is None:
            xycfn = ray_bundle.lengths
        else:
            xycfn = xycfn_new

        ray_sh = ray_bundle.origins.shape

        # Combine and sort all intersections
        # Sort by intersections along the ray coordinates
        batch_idx = (
            torch.linspace(0, ray_sh[0] - 1, ray_sh[0], dtype=int)[None]
                .repeat(ray_sh[1], 1)
                .transpose(1, 0)
        )
        batch_idx = batch_idx[..., None].repeat(1, 1, lengths.shape[-1])
        batch_ray_idx = (
            torch.linspace(0, ray_sh[1] - 1, ray_sh[1], dtype=int)[:, None]
                .repeat(1, ray_sh[0])
                .transpose(1, 0)
        )
        batch_ray_idx = batch_ray_idx[..., None].repeat(1, 1, lengths.shape[-1])
        lengths, length_sorted_idx = torch.sort(lengths, dim=-1)

        sorted_mask = tuple([batch_idx, batch_ray_idx, length_sorted_idx])

        sort_intersections = length_sorted_idx[intersection_mask[:-2]].flatten(
            0, -2
        ) == intersection_mask[2].flatten()[:, None].repeat(1, lengths.shape[-1])
        sort_intersections = torch.where(sort_intersections)

        sorted_intersection_mask = list(intersection_mask)
        sorted_intersection_mask[2] = sort_intersections[1].reshape(-1, self._n_samples)

        ray_bundle = ray_bundle._replace(lengths=lengths, xys=xycfn[sorted_mask])

        sorted_intersection_mask = tuple(
            [v.flatten() for v in sorted_intersection_mask]
        )

        return ray_bundle, sorted_intersection_mask


class BoxRaysampler(nn.Module):
    def __init__(
            self,
            n_pts_box: int,
    ):
        super().__init__()
        self._n_samples = n_pts_box

    def forward(self, box_in: torch.Tensor, box_out: torch.Tensor, **kwargs):

        if box_in.dim() < 2:
            box_in = box_in[:, None]
        if box_out.dim() < 2:
            box_out = box_out[:, None]

        # Potential importance sampling etc...
        # Uniform sampling between near and far
        z_vals = (
                box_in
                + (box_out - box_in)
                * torch.linspace(
            0.0, 1.0, self._n_samples, device=box_in.device, dtype=box_in.dtype
        )[None, :]
        )

        return z_vals