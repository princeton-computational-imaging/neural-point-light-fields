import numpy as np
import torch
import torch.nn as nn
from pytorch3d.renderer.implicit.utils import RayBundle
from pytorch3d.transforms import Transform3d
from pytorch3d.transforms.so3 import so3_log_map, so3_exponential_map


def join_list(x):
    return [inner for outer in x for inner in outer]


class RayIntersection(nn.Module):
    """
    Module to compute Ray intersections
    """

    def __init__(self,
                 n_samples: int,
                 chunk_size: int):
        super().__init__()
        self._n_samples = n_samples
        self._chunk_size = int(chunk_size)

    def forward(self,
                ray_bundle: RayBundle,
                scene,
                intersections: torch.Tensor = None,
                **kwargs
                ):
        """
        Takes rays [ray_origin, ray_direction, frame_id], nodes to intersect and a scene as an Input.
        Outputs intersection points in ray coordinates, [z_val, node_scene_id, frame_id]
        """
        # TODO: Only compute required intersection with plane/boxes/etc. in test time

        # return intersection_pts
        pass


class RayBoxIntersection(RayIntersection):
    def __init__(self,
                 box_nodes: dict,
                 chunk_size: int,
                 n_intersections_box: int=2):

        super().__init__(n_samples=n_intersections_box,
                         chunk_size=chunk_size)

        self._box_nodes = [node.scene_idx for node in box_nodes.values()]

    def forward(self,
                origins: torch.Tensor,
                directions: torch.Tensor,
                box_bounds: torch.Tensor = torch.tensor([[-1, -1, -1], [1, 1, 1]]),
                transformation: Transform3d = None,
                output_points: bool = False,
                **kwargs,
                ):

        if transformation is not None:
            origins = transformation.transform_points(origins)
            directions = transformation.transform_points(directions)

        if directions.dim() == 2:
            n_batch = 1
            n_rays_batch = directions.shape[0]
        elif directions.dim() == 3:
            ray_d_sh = directions.shape
            n_batch = ray_d_sh[0]
            n_rays_batch = ray_d_sh[1]
            directions = directions.flatten(0, -2)
            origins = origins.flatten(0, -2)
        else:
            ValueError("Ray directions are of dimesion {}, but must be of dimension 2 or 3.".format(directions.dim()))

        if origins.dim() < directions.dim():
            origins = origins.expand(n_batch, n_rays_batch, 3)
            origins = origins.flatten(0, -2)
        else:
            if origins.shape != directions.shape:
                ValueError()

        box_bounds = box_bounds.to(origins.device)
        # Perform ray-aabb intersection and return in and out without chunks
        inv_d = torch.reciprocal(directions)

        t_min = (box_bounds[0] - origins) * inv_d
        t_max = (box_bounds[1] - origins) * inv_d

        t0 = torch.minimum(t_min, t_max)
        t1 = torch.maximum(t_min, t_max)
        t_near = torch.maximum(torch.maximum(t0[..., 0], t0[..., 1]), t0[..., 2])
        t_far = torch.minimum(torch.minimum(t1[..., 0], t1[..., 1]), t1[..., 2])
        # Check if rays are inside boxes
        intersection_idx = torch.where(t_far > t_near)
        # Check that boxes are in front of the ray origin
        intersection_idx = intersection_idx[0][t_far[intersection_idx] > 0]
        if not len(intersection_idx) == 0:
            z_in = t_near[intersection_idx]
            z_out = t_far[intersection_idx]

            # Reindex for [n_batches, ....] and sort again
            batch_idx = torch.floor(intersection_idx / n_rays_batch).to(torch.int64)
            intersection_idx = intersection_idx % n_rays_batch
            # intersection_idx, new_sort = torch.sort(intersection_idx)
            intersection_mask = tuple([batch_idx, intersection_idx])

            if not output_points:
                return [z_in, z_out], intersection_mask
            else:
                pts_in = origins.view(-1, n_rays_batch, 3)[intersection_idx] + \
                         directions.view(-1, n_rays_batch, 3)[intersection_idx] * z_in[:, None]
                pts_out = origins.view(-1, n_rays_batch, 3)[intersection_idx] + \
                          directions.view(-1, n_rays_batch, 3)[intersection_idx] * z_out[:, None]

                return [z_in, z_out, pts_in, pts_out], intersection_mask

        else:
            if output_points:
                return [None, None, None, None], None
            else:
                return [None, None], None


class RaySphereIntersection(RayIntersection):
    def __init__(self,
                 n_samples_box: int,
                 chunk_size: int,):
        super().__init__(n_samples=n_samples_box)

    def forward(self,
                ray_bundle: RayBundle,
                scene,
                intersections: torch.Tensor = None,
                **kwargs):
        intersection_pts = []
        return intersection_pts


class RayPlaneIntersection(RayIntersection):
    def __init__(self,
                 n_planes: int,
                 near: float,
                 far: float,
                 chunk_size: int,
                 camera_poses: list,
                 background_trafos: torch.Tensor,
                 transient_background: bool=False):
        super().__init__(n_samples=n_planes,
                         chunk_size=chunk_size)

        self._planes_n = {}
        self._planes_p = {}
        self._plane_delta = {}
        self._near = {}
        self._far = {}
        self._transient_background = False

        for key, val in background_trafos.items():
            if self._transient_background:
                self._near[key] = torch.as_tensor(near[key])
                self._far[key] = torch.as_tensor(far[key])

            else:
                global_trafo = val
                near_k = torch.as_tensor(near[key])
                far_k = torch.as_tensor(far[key])

                self._planes_n[key] = global_trafo[:3, 2]

                all_camera_poses = []
                for edge_dict in camera_poses.values():
                    for edge in edge_dict.values():
                        if len(edge) == 0:
                            continue
                        if edge[0].parent == key:
                            all_camera_poses.append(edge[0].translation)

                all_camera_poses = torch.cat(all_camera_poses)
                n_cameras = 2
                # len(camera_poses)
                # assert n_cameras == 2
                n_cam_poses = len(all_camera_poses)

                max_pose_dist = torch.norm(all_camera_poses[-1] - all_camera_poses[0])

                # TODO: shouldn't this be int(n_cam_poses / n_cameras) +1
                end = int(n_cam_poses / n_cameras) + 1
                pose_dist = (all_camera_poses[1:end] - all_camera_poses[:end-1])
                pose_dist = torch.norm(pose_dist, dim=1).max()
                planes_p = global_trafo[:3, -1] + near_k * self._planes_n[key]

                self._plane_delta[key] = (far_k - near_k) / (self._n_samples - 1)

                poses_per_plane = int(((far_k - near_k) / self._n_samples) / pose_dist)
                if poses_per_plane != 0:
                    add_planes = int(np.ceil((n_cam_poses/n_cameras) / poses_per_plane))
                else:
                    add_planes = 1
                id_planes = torch.linspace(0, self._n_samples + add_planes - 1, self._n_samples+ add_planes, dtype=int)

                self._planes_p[key] = planes_p + (id_planes * self._plane_delta[key])[:, None] * self._planes_n[key]
                far_k = near_k + self._plane_delta[key] * (id_planes[-1] + add_planes)

                self._near[key] = torch.as_tensor(near_k)
                self._far[key] = torch.as_tensor(far_k)

    def forward(self,
                ray_bundle: RayBundle,
                scene,
                intersections: torch.Tensor = None,
                obj_obly: bool=False,
                **kwargs):
        """ Ray-Plane intersection for given planes in the scenes

            Args:
                rays: ray origin and directions
                planes: first plane position, plane normal and distance between planes
                id_planes: ids of used planes
                near: distance between camera pose and first intersecting plane

            Returns:
                pts: [N_rays, N_samples+N_importance] - intersection points of rays and selected planes
                z_vals: integration step along each ray for the respective points
            """
        if not obj_obly and not self._transient_background:
            node_id = 0

            # TODO: Compare with outputs from old method
            # Extract ray and plane definitions
            device = ray_bundle.origins.device
            N_rays = np.prod(ray_bundle.origins.shape[:-1])

            # Get amount of all planes
            rays_sh = list(ray_bundle.origins.shape)
            xys_sh = list(ray_bundle.xys.shape)

            # Flatten ray origins and directions
            all_origs = ray_bundle.origins.flatten(0, -2)
            all_dirs = ray_bundle.directions.flatten(0, -2)
            xycfn = ray_bundle.xys.flatten(0,-2)

            if len(xycfn) != len(all_origs):
                Warning("Please check that global sampler is executed before the local sampler!")

            # TODO: Initilaize planes with right dtype float32
            # TODO: Check run time for multiple scene implementation
            d_origin_planes = torch.zeros([self._n_samples, len(xycfn)], device=device)

            for n_idx in xycfn[:,-1].unique():
                n_i_mask = torch.where(xycfn[:, -1] == n_idx)
                background_mask = [
                    torch.linspace(0, self._n_samples - 1, self._n_samples, dtype=torch.int64)[:, None].repeat(1,len(n_i_mask[0])),
                    n_i_mask[0][None].repeat(self._n_samples, 1)]

                n_idx = int(n_idx)
                origs = all_origs[n_i_mask]
                dirs = all_dirs[n_i_mask]

                p = self._planes_p[n_idx].to(device=origs.device, dtype=origs.dtype)
                n = self._planes_n[n_idx].to(device=origs.device, dtype=origs.dtype)
                near = self._near[n_idx].to(device=origs.device, dtype=origs.dtype)
                delta = self._plane_delta[n_idx]

                if len(p) > self._n_samples:
                    # import matplotlib.pyplot as plt
                    #
                    # plt.scatter(origs[:, 0], origs[:, 2])
                    # plt.scatter(p[:, 0], p[:, 2])
                    # plt.axis('equal')
                    # Just get the intersections with self._n_planes - 1 planes in front of the camera and the last plane
                    d_p0_orig = torch.matmul(p[0] - origs - 1e-3, n)
                    d_p0_orig = torch.maximum(-d_p0_orig, -near)
                    start_idx = torch.ceil((d_p0_orig + near) / delta).to(dtype=torch.int64)
                    plane_idx = start_idx + torch.linspace(0, self._n_samples - 2, self._n_samples - 1, dtype=int, device=near.device)[:, None]
                    plane_idx = torch.cat([plane_idx, torch.full([1, len(origs)], len(p) - 1, device=near.device)])
                    p = p[plane_idx]
                else:
                    p = p[:, None, :]

                d_origin_planes_i = p - origs
                d_origin_planes_i = torch.matmul(d_origin_planes_i, n)
                d_origin_planes_i = d_origin_planes_i / torch.matmul(dirs, n)
                # TODO: Include check that validity here (if everything is positive)
                d_origin_planes[background_mask] = d_origin_planes_i

            rays_sh.insert(-1, self._n_samples)
            lengths = d_origin_planes.transpose(1, 0).reshape(rays_sh[:-1])
        else:
            if not self._transient_background:
                device = ray_bundle.lengths.device
                near = min(list(self._near.values()))
                far = max(list(self._far.values()))
                z = torch.linspace(near, far, self._n_samples)
                lengths = torch.ones(ray_bundle.lengths.shape[:-1])[..., None].repeat(1, 1, self._n_samples).to(device)
                lengths *= z.to(device)
            else:
                device = ray_bundle.lengths.device
                far = max(list(self._far.values()))
                lengths = torch.ones(ray_bundle.lengths.shape[:-1])[..., None].repeat(1, 1, self._n_samples).to(
                    device)
                lengths *= torch.tensor([far]).to(device)

        return lengths