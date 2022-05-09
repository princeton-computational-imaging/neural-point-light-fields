from typing import List, Dict
import copy

import numpy as np
import torch, tqdm, os
import torch.nn as nn
from .pointsampler import BackgroundRaysampler, ObjectRaysampler, PointDistanceRaysamplerTorch, PointDistanceRaysamplerNP
from pytorch3d.renderer.implicit.utils import RayBundle
from pytorch3d.transforms import Transform3d, Translate, Rotate, Scale
import matplotlib.pyplot as plt

from ... import utils as ut
from haven import haven_utils as hu
import time


def join_list(x):
    return [inner for outer in x for inner in outer]


class Raysampler:
    """
    n_pts_background
    n_pts_object
    camera_nodes to initialize static and object sampler for each camera
    near clip for static sampler
    far clip for static sampler
    """

    def __init__(
        self,
        global_sampler,
        local_sampler,
        scene,
        reference_frame: str = 'world',
        n_rays_per_image: int = 1024,
        use_gt_masks=False,
        exp_dict=False,
        additional_ray_information=False,
    ):
        self.use_gt_masks = use_gt_masks
        self.cameras = scene.nodes["camera"]
        self.exp_dict = exp_dict
        self._global_sampler = global_sampler
        self._local_sampler = local_sampler

        self._ray_cache = {}
        self._mask_cache = {}
        self._pts_cache = {}
        self._n_rays_per_image = n_rays_per_image
        self._n_rays_per_image_train = n_rays_per_image

        self._reference_frame = reference_frame

        self.additional_ray_information = additional_ray_information

    @torch.no_grad()
    def get_intersections_only(self, ray_bundle, scene, fi, ci, faster=False, save_intersections=False):
        n_pixels = ray_bundle.lengths.shape[1]
        fname_end = hu.hash_dict({'0':
                                  scene.frames[fi].images[ci]}
                                )
        fname = os.path.join('/mnt/public/datasets/tmp', fname_end+'.pkl')
        assert fname != scene.frames[fi].images[ci]
        
        if not os.path.exists(fname):
            # Select only intersecting rays
            ray_bundle = RayBundle(
                *[
                    v.view([1] + [v.shape[0] * v.shape[1]] + list(v.shape[2:]))
                    for v in ray_bundle
                ]
            )

            if faster:
                flat_segmentation_mask = scene.frames[fi].load_mask(ci).flatten()
                ray_start = np.where(flat_segmentation_mask > 0.0)[0].min()
                ray_end = np.where(flat_segmentation_mask > 0.0)[0].max()

                reduced_rb = RayBundle(*[v[:, ray_start:ray_end] for v in ray_bundle])
                intersection_mask = self._local_sampler._precache_mask(reduced_rb, scene)
                if intersection_mask is None:
                    return None
                intersection_mask = tuple(
                    [intersection_mask[0], intersection_mask[1] + ray_start]
                )

            else:
                intersection_mask = self._local_sampler._precache_mask(ray_bundle, scene)
            if save_intersections and os.path.exists('/mnt/public/datasets/tmp'):
                hu.save_pkl(fname, intersection_mask)

        else:
            intersection_mask = hu.load_pkl(fname)

        if intersection_mask is None:
            return None

        n_intersects = ray_bundle.lengths[intersection_mask].shape[0]
        if n_intersects < self._n_rays_per_image:
            t2 = np.setdiff1d(np.arange(n_pixels), intersection_mask[1].numpy() )
            t3 = torch.from_numpy(np.random.choice(t2, self._n_rays_per_image-n_intersects))
            intersection_mask = [torch.zeros(self._n_rays_per_image).long(), 
                                 torch.cat((intersection_mask[1], t3), dim=0).long()] 
      
        ray_bundle = RayBundle(
            *[
                v[intersection_mask].view(
                    [1] + [len(intersection_mask[0])] + list(v.shape[2:])
                )
                for v in ray_bundle
            ]
        )
        return ray_bundle

    def forward(
            self,
            scene=None,
            chunk_idx: int = 0,
            frame_idx: list = None,
            camera_idx: list = None,
            obj_only: bool = True,
            intersections_only: bool = False,
            random_rays: bool = False,
            validation: bool = False,
            EPI: bool = False,
            epi_row: bool = None,
    ):
        """
        Args:
            scene: A Scene
            chunksize: ...
            chunk_idx: ....
            frame_idx: A list of `N` frames from which the rays are evaluated.
            camera_idx: A list of `N` cameras from which the rays respective frame in frame_idx.
        Returns:
            A named tuple `RayBundle` with the following fields:
                origins: A tensor of shape
                    `(batch_size, n_rays_per_image, 3)`
                    denoting the locations of ray origins in the world coordinates.
                directions: A tensor of shape
                    `(batch_size, n_rays_per_image, 3)`
                    denoting the directions of each ray in the world coordinates.
                lengths: A tensor of shape
                    `(batch_size, n_rays_per_image, n_pts_per_ray)`
                    containing the z-coordinate (=depth) of each ray in world units.
                xys: A tensor of shape
                    `(batch_size, n_rays_per_image, 2)`
                    containing the 2D image coordinates of each ray.
        """
        # assert obj_only == True
        # if intersections_only:
        #     obj_only = True
        # Get device from trainable components
        device = "cpu"

        # 1. Get full ray bundle
        full_ray_bundle = []
        fi = frame_idx
        ci = camera_idx
        # Precached case
        if len(self._ray_cache) > 0:
            rb = self._ray_cache[(fi, ci)]

        # Not Precached case
        else:
            ## Get all transformations to all cameras
            edges2cams = scene.get_all_edges_to_cameras([ci], [fi])
            cam_nodes = {}
            for c in [ci]:
                cam_nodes[c] = self.cameras[c]

            rb = self._get_rays(cameras=cam_nodes, edges=edges2cams, device=device, scene=scene, optimize_cam=scene.recalibrate, EPI=EPI,
                                epi_row=epi_row)

        # get only intersections
        if (
            intersections_only
            and obj_only
            and self.exp_dict.get("use_intersections", True)
        ):
            # get intersections to "cubes for instance"
            rb = self.get_intersections_only(rb, scene, fi, ci, faster=False, 
                            save_intersections=self.exp_dict.get("save_intersections", False))

        if rb is not None:
            full_ray_bundle += [rb]
        else:
            print("warning - rb is empty")
        if rb is None:
            return None
        # full_ray_bundle = stack_rays(full_ray_bundle)
        full_ray_bundle = rb
        batch_size, n_pixels = get_batch_size_n_pixels(full_ray_bundle)

        # 2. Select Rays
        if intersections_only or random_rays:
            # Select Random Rays
            sel_rays = torch.randperm(n_pixels, device=device)[
                : self._n_rays_per_image_train
            ]

            # ut.extract_patch(rb.xys, scene.frames[fi].H, scene.frames[fi].W)
        else:
            # Validation step
            assert batch_size == 1
            chunksize = n_pixels * batch_size
            start = chunk_idx * chunksize * batch_size
            end = min(start + chunksize, n_pixels)
            sel_rays = torch.arange(
                start,
                end,
                dtype=torch.long,
                device=full_ray_bundle.lengths.device,
            )

        # Index the rays with the selected indices
        ray_bundle = select_rays(sel_rays, full_ray_bundle)

        # 3. Compute intersections with all scene components
        # ray_bundle = self._global_sampler(ray_bundle, scene, obj_only=obj_only)
        # ray_bundle, local_pts_idx, local_pts, local_dirs = self._local_sampler(
        #     ray_bundle, scene
        # )
        # orig = ray_bundle.origins.clone()
        # dir = ray_bundle.directions.clone()
        # ray_bundle = ray_bundle._replace(origins=ray_bundle.origins.detach(), directions=ray_bundle.directions.detach())
        if not obj_only:
            ray_bundle = self._global_sampler(ray_bundle, scene=scene, obj_only=obj_only)
            ray_bundle, local_pts_idx, local_pts, local_dirs = self._local_sampler(
                ray_bundle, scene, validation=validation,
            )
        else:
            ray_bundle, local_pts_idx, local_pts, local_dirs = self._local_sampler(
                ray_bundle, scene, validation=validation,
            )
            if local_pts_idx is None:
                return None
            # Reduce number of elements in the full ray bundle
            first_intersection_id = torch.where(ray_bundle.lengths > 0.0)[2].min()
            ray_bundle = ray_bundle._replace(
                lengths=ray_bundle.lengths[..., first_intersection_id:],
                xys=ray_bundle.xys[..., first_intersection_id:, :],
            )
            local_pts_idx = torch.stack(local_pts_idx)
            local_pts_idx[2] = local_pts_idx[2] - first_intersection_id
            local_pts_idx = tuple(local_pts_idx)

        # TODO: From here move out of raysampler and into __getitem__
        if local_pts_idx is None:
            raise ValueError("do the old version")

        ray_dict = {
            "ray_bundle": ray_bundle,
            "pts": local_pts,
            "pts_idx": local_pts_idx,
            "pts_dirs": local_dirs
        }

        if self.additional_ray_information:
            ray_dict.update(
                self.extract_additional_ray_information(
                    scene, ray_bundle, local_pts_idx, local_pts, local_dirs, frame_idx
                )
            )

        return ray_dict

    def _get_rays(
        self,
        cameras: dict,
        edges: dict,
        device: str,
        scene=None,
        optimize_cam: bool = False,
        EPI=False,
        epi_row=None,
    ):

        origins = []
        directions = []
        xycf = []

        k, cam = list(cameras.items())[0]
        c_idx = cam.scene_idx
        f_idx = []
        background_node = []
        trafo = None

        frame_id, frame_edges = list(edges[k].items())[0]
        f_idx.append(frame_id)
        background_node.append(frame_edges[0].parent)
        for ed in frame_edges:
            openGL2dataset = Rotate(cam.R, device=device)

            if self._reference_frame == 'lidar':
                # TODO: Allow training on all point clouds
                top_lidar_mask = [True if v.name == 'TOP' else False for k, v in scene.nodes['lidar'].items()]
                top_lidar_id = int(np.where(np.array(top_lidar_mask))[0])

                lidar_idx = list(scene.nodes['lidar'])
                lidar_edges = scene.frames[frame_id].get_edge_by_child_idx(lidar_idx)
                lidar_ed = [edges for edges in lidar_edges if len(edges) > 0][top_lidar_id][0]

                li2cam = torch.eye(4, device=device)[None]

                li2cam[0, 3, :3] = lidar_ed.translation - ed.translation
                li2cam[0, :3, :3] = ed.getRotation_c2p().compose(lidar_ed.getRotation_p2c()).get_matrix()[:, :3, :3]

                if trafo is not None:
                    trafo = trafo.stack(Transform3d(matrix=li2cam)).to(device=device)
                else:
                    trafo = Transform3d(matrix=li2cam)
            elif self._reference_frame == 'vehicle':
                veh2wo = scene.frames[frame_id].global_transformation
                wo2veh = np.concatenate(
                    [veh2wo[:3, :3].T,
                     veh2wo[:3, :3].T.dot(-veh2wo[:3, 3])[:, None]],
                    axis=1
                )
                wo2veh = np.concatenate([wo2veh, np.array([[0., 0., 0., 1.]])])

                if not optimize_cam:
                    cam2wo = ed.get_transformation_c2p().cpu().get_matrix()[0].T.detach()
                    cam2veh = wo2veh.dot(cam2wo.numpy())
                    cam2veh = Transform3d(matrix=torch.tensor(cam2veh.T, device=device, dtype=torch.float32))
                else:
                    cam2wo = ed.get_transformation_c2p(device='cpu').get_matrix()[0].T
                    cam2veh = torch.matmul(torch.tensor(wo2veh, dtype=cam2wo.dtype), cam2wo)
                    cam2veh = Transform3d(matrix=cam2veh.T)

                if trafo is not None:
                    trafo = trafo.stack(cam2veh).to(device=device)
                else:
                    trafo = cam2veh

            elif self._reference_frame == 'world':
                if trafo is not None:
                    trafo = trafo.stack(ed.get_transformation_c2p().to(device=device)).to(
                        device=device
                    )
                else:
                    trafo = ed.get_transformation_c2p().to(device=device)
            else:
                if trafo is not None:
                    trafo = trafo.stack(ed.get_transformation_c2p().to(device=device)).to(
                        device=device
                    )
                else:
                    trafo = ed.get_transformation_c2p().to(device=device)

        W = cam.intrinsics.W
        H = cam.intrinsics.H
        n_fr = len(f_idx)
        # TODO: Change for only a single focal length in camera node
        focal = cam.intrinsics.f_x.to(device)

        i, j = torch.meshgrid(
            torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H)
        )  # pytorch's meshgrid has indexing='ij'
        i = i.t().to(device)
        j = j.t().to(device)

        f = torch.tensor(f_idx, device=device)[:, None, None].repeat(1, H, W)
        c = torch.full((n_fr, H, W), c_idx, device=device)
        n = torch.tensor(background_node, device=device)[:, None, None].repeat(1, H, W)
        new_xycf = (
            torch.stack(
                [i[None].repeat(n_fr, 1, 1), j[None].repeat(n_fr, 1, 1), c, f, n],
                dim=-1,
            )
            .to(torch.int32)
            .flatten(1, 2)
        )

        # Create ray directions in an OpenGL conform camera coordinate frame
        ray_d = torch.stack(
            [
                (i - W * 0.5) / focal,
                -(j - H * 0.5) / focal,
                -torch.ones_like(i, device=device),
            ],
            -1,
        )

        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        ray_o = trafo.get_matrix()[:, -1, None, :3]
        # Rotate ray directions from camera frame to the world frame
        ray_d = (
            openGL2dataset
            .compose(trafo.translate(-ray_o[:, 0, :]))
            .transform_points(ray_d.reshape(1, -1, 3))
        )
        ray_o = ray_o.expand(ray_d.shape)
        # Normalize ray directions
        ray_d = (1 / torch.norm(ray_d, dim=-1))[..., None] * ray_d

        directions.append(ray_d)
        origins.append(ray_o)
        xycf.append(new_xycf)

        origins = torch.cat(origins).view([1, -1, 3])
        dirs = torch.cat(directions).view([1, -1, 3])
        xycfn = torch.cat(xycf).view([1, -1, 5])
        lengths = torch.zeros(
            [origins.shape[0], origins.shape[1], 0], device=origins[0].device
        ).view([1, dirs.shape[1], 0])

        if EPI:
            print("RENDERING EPI AND NOT FRAME")
            new_H = 101
            scene.frames[frame_id].H = new_H
            new_H = scene.frames[frame_id].H

            epi_dirs = dirs.reshape(1, H, W, 3)
            epi_origs = origins.reshape(1, H, W, 3)

            focus_distance = torch.tensor([7.])

            pixel_width = torch.norm((epi_dirs[..., 1:, :] * focal / 1000) - (epi_dirs[..., :-1, :] * focal / 1000),
                                     dim=-1).mean()
            sensor_width = pixel_width * W
            x_movement = torch.tensor([1.])

            epi_deg = (-torch.atan2(x_movement, focus_distance)).numpy()
            # np.rad2deg(epi_deg)

            center_orig = epi_origs[:, epi_row, W // 2]

            movement = x_movement * torch.tensor([1., 0., 0.], device=ray_o.device, dtype=ray_o.dtype)
            scan_line_o = movement[None] * torch.linspace(-1., 1., new_H, device=ray_o.device, dtype=ray_o.dtype)[:, None]
            scan_line_o = trafo.transform_points(scan_line_o[:, None]) - center_orig

            scan_line_d = epi_dirs[:, epi_row]

            new_origs = center_orig + scan_line_o
            new_origs = new_origs.repeat(1, W, 1)

            rots = [torch.tensor([[torch.cos(ang), -torch.sin(ang), 0.],
                                  [torch.sin(ang), torch.cos(ang), 0.],
                                  [0., 0., 1.], ])
                    for ang in torch.linspace(-float(epi_deg), float(epi_deg), new_H)]

            new_dirs = torch.stack([torch.matmul(rot, scan_line_d.squeeze().T).T for rot in rots])[None]

            test_view = False
            if test_view:
                new_dirs = torch.matmul(rots[0], dirs[0].T).T[None]
                new_origs = origins

            origins = new_origs.view(1, -1, 3)
            dirs = new_dirs.view(1, -1, 3)
            lengths = lengths[:, :dirs.shape[1]]
            xycfn = xycfn[:, :dirs.shape[1]]

        ray_bundle = RayBundle(origins, dirs, lengths, xycfn)
        return ray_bundle

    def extract_additional_ray_information(
        self, scene, ray_bundle, local_pts_idx, local_pts, local_dirs, frame_idx
    ):
        xycfn_objs = ray_bundle.xys[local_pts_idx]
        obj_ids = xycfn_objs.unique()
        obj_ids_expected = set(
            scene.frames[frame_idx].get_objects_ids()
            + [-1, scene.frames[frame_idx].background_node]
        )

        for o in obj_ids:
            assert int(o) in obj_ids_expected

        class_idx, object_idx = get_object_node_information(scene, obj_ids=obj_ids)
        # Car object class only
        node_idx = class_idx[0]
        assert len(class_idx.unique()) == 1

        # Get all object nodes that are connected to the object class at node_idx
        class2obj_idx = torch.where(class_idx == node_idx)

        # Get the index to all points that intersect with a node of this class by checking for the node in all intersection points
        intersections_obj_idx = [
            torch.where(xycfn_objs == i)[0] for i in object_idx[class2obj_idx]
        ]
        model_intersection_idx = torch.cat(intersections_obj_idx)

        # Get the idx of rays intersecting this model node in the ray bundle
        local_pts_model_idx = list(
            idx[model_intersection_idx] for idx in local_pts_idx[:-1]
        )

        # Get the intersection points and ray directions in the local coordinate frames
        pts = local_pts.reshape(-1, 3)[model_intersection_idx]
        dirs = local_dirs.reshape(-1, 3)[model_intersection_idx]
        return {
            "pts": pts,
            "dirs": dirs,
            "locs": None,
            "object_idx": object_idx,
            "object_len": [len(idx) for idx in intersections_obj_idx],
            "local_pts_model_idx": local_pts_model_idx,
            "node_class_idx": node_idx,
        }

    def join_ray_bundle_batch(self, ray_bundle_batch, locap_pts_idx_batch, device):
        local_pts_idx = None
        if len(ray_bundle_batch) > 1:
            if ray_bundle_batch[0].xys.dim() == 3:
                ray_bundle = RayBundle(*[torch.cat([rb[k] for rb in ray_bundle_batch]).to(device) for k in range(4)])
            else:
                n_intersections_ls = [rb.lengths.shape[-1] for rb in ray_bundle_batch]
                n_intersections = np.array(n_intersections_ls)
                max_intersections = max(n_intersections)
                paddings = max_intersections - n_intersections

                xycfn_placeholder = torch.stack(
                    [rb.xys[..., 0, None, :] for rb in ray_bundle_batch]
                )
                xycfn_placeholder[..., -1] = -1

                lengths_stacked = torch.cat(
                    [
                        torch.cat(
                            [
                                torch.zeros(
                                    size=[1, self._n_rays_per_image, pad], device=device
                                ),
                                rb.lengths.to(device),
                            ],
                            dim=2,
                        )
                        for pad, rb in zip(paddings, ray_bundle_batch)
                    ],
                    dim=0,
                )

                xycfn_placeholder = torch.stack(
                    [rb.xys[..., 0, None, :].to(device) for rb in ray_bundle_batch]
                )
                xycfn_placeholder[..., -1] = -1

                xycfn_stacked = torch.cat(
                    [
                        torch.cat(
                            [xycfn_plh.repeat(1, 1, pad, 1), rb.xys.to(device)], dim=2
                        )
                        for pad, rb, xycfn_plh in zip(
                            paddings, ray_bundle_batch, xycfn_placeholder
                        )
                    ],
                    dim=0,
                )

                origins_stacked = torch.cat([rb.origins for rb in ray_bundle_batch]).to(
                    device
                )
                directions_stacked = torch.cat(
                    [rb.directions for rb in ray_bundle_batch]
                ).to(device)

                local_pts_idx = [
                    torch.cat([v[0] + l for l, v in enumerate(locap_pts_idx_batch)]),
                    torch.cat([v[1] for l, v in enumerate(locap_pts_idx_batch)]),
                    torch.cat(
                        [v[2] + pad for pad, v in zip(paddings, locap_pts_idx_batch)]
                    ),
                ]
                ray_bundle = RayBundle(
                    origins=origins_stacked,
                    directions=directions_stacked,
                    xys=xycfn_stacked,
                    lengths=lengths_stacked,
                )
        else:
            ray_bundle = RayBundle(
                origins=ray_bundle_batch[0].origins.to(device),
                directions=ray_bundle_batch[0].directions.to(device),
                xys=ray_bundle_batch[0].xys.to(device),
                lengths=ray_bundle_batch[0].lengths.to(device),
            )
            if locap_pts_idx_batch is not None:
                local_pts_idx = locap_pts_idx_batch[0]

        return ray_bundle, local_pts_idx


def get_batch_size_n_pixels(full_ray_bundle):
    batch_size = full_ray_bundle.directions.shape[0]
    n_pixels = full_ray_bundle.directions.shape[:-1].numel()
    return batch_size, n_pixels


def stack_rays(full_ray_bundle):
    full_ray_bundle = RayBundle(
        *[
            torch.cat([ray_bundle[i] for ray_bundle in full_ray_bundle], dim=1)
            for i in range(4)
        ]
    )
    return full_ray_bundle


def select_rays(sel_rays, full_ray_bundle):
    batch_size, n_pixels = get_batch_size_n_pixels(full_ray_bundle)

    ray_bundle = RayBundle(
        *[
            v.view([n_pixels] + list(v.shape[2:]))[sel_rays]
            .view([1, sel_rays.numel()] + list(v.shape[2:]))
            .to(v.device)
            for v in full_ray_bundle
        ]
    )
    return ray_bundle


def get_object_node_information(scene, obj_ids):
    """

    Args:
        scene:
        obj_ids:
        relevant_xycfn:
    Returns:
        class_idx:
        object_idx:
        object_latent:
    """
    # Get all latent codes and the index to the respective models
    class_idx = []
    object_idx = []
    # Run time n_class
    for n in obj_ids:
        obj_node = scene.nodes[int(n)]["node"]
        assert scene.nodes[int(n)]["node_type"] == "scene_object"
        class_node = scene.type2class[obj_node.object_class_type_idx]

        class_idx.append(class_node.scene_idx)
        object_idx.append(n)

    class_idx = torch.tensor(class_idx)
    object_idx = torch.tensor(object_idx)

    return class_idx, object_idx


class NeuralSceneRaysampler(Raysampler):
    def __init__(
            self,
            scene=None,
            n_pts_background: int = 6,
            n_pts_object: int = 7,
            n_rays_per_image: int = 1024,
            exp_dict: Dict = {},
    ):
        global_sampler = BackgroundRaysampler(
            camera_poses=scene.get_all_edges_to_cameras(list(scene.nodes['camera'].keys())),
            n_pts_background=n_pts_background,
            background_nodes=scene.nodes["background"],
        )
        local_sampler = ObjectRaysampler(
            class_nodes=scene.nodes["object_class"],
            n_pts_object=n_pts_object,
        )

        super(NeuralSceneRaysampler, self).__init__(global_sampler=global_sampler,
                                                    local_sampler=local_sampler,
                                                    scene=scene,
                                                    reference_frame='world',
                                                    n_rays_per_image=n_rays_per_image,
                                                    use_gt_masks=False,
                                                    exp_dict=exp_dict,
                                                    additional_ray_information=True,
                                                    )


class PointLightFieldSampler(Raysampler):
    def __init__(self,
                 scene,
                 lightfield_config: Dict,
                 n_rays_per_image: int = 1024,
                 reference_frame: str = 'lidar',
                 exp_dict: Dict = {},
                 ):
        global_sampler = lambda ray_bundle, obj_only=False, **kwargs: ray_bundle

        if not lightfield_config.get('merge_pcd', True):
            lightfield_config.get('num_merged_frames', 1)

        if scene.scene_descriptor.get('pt_cloud_fix', False):
            n_merged_fr = lightfield_config.get('num_merged_frames', 20)
            reference_frame = 'vehicle'
        elif scene.dataset.type == "waymo" and scene.dataset.scene_id == [0, 2]:
            n_merged_fr = lightfield_config.get('num_merged_frames', 5)
        elif scene.dataset.type == "waymo":
            n_merged_fr = lightfield_config.get('num_merged_frames', 10)
        else:
            n_merged_fr = lightfield_config.get('num_merged_frames', 40)

        if not (lightfield_config.get('optimize_cam', False) or lightfield_config.get('torch_sampler', False)):
            local_sampler = PointDistanceRaysamplerNP(n_pts=lightfield_config['n_sample_pts'],
                                                      k_closest=lightfield_config['k_closest'],
                                                      point_chunk_size=1e12,
                                                      merge_pcd=lightfield_config.get('merge_pcd', False),
                                                      n_frames_merged=n_merged_fr,
                                                      augment_point_cloud=lightfield_config.get('augment_frame_order', False),
                                                      new_encoding=lightfield_config.get('new_enc', False),
                                                      pt_caching=exp_dict.get('pt_cache', False),
                                                      )
        else:
            local_sampler = PointDistanceRaysamplerTorch(n_pts=lightfield_config['n_sample_pts'],
                                                         k_closest=lightfield_config['k_closest'],
                                                         point_chunk_size=1e12,
                                                         merge_pcd=lightfield_config.get('merge_pcd', False),
                                                         n_frames_merged=n_merged_fr,
                                                         augment_point_cloud=lightfield_config.get('augment_frame_order', False),
                                                         new_encoding=lightfield_config.get('new_enc', False),
                                                         pt_caching=exp_dict.get('pt_cache', False),
                                                         )

        super(PointLightFieldSampler, self).__init__(global_sampler=global_sampler,
                                                     local_sampler=local_sampler,
                                                     scene=scene,
                                                     reference_frame=reference_frame,
                                                     n_rays_per_image=n_rays_per_image,
                                                     n_pts_object=0,
                                                     exp_dict=exp_dict,
                                                     )
        