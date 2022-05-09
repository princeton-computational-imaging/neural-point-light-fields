from typing import List

import numpy as np
import torch
import torch.nn as nn
from ..scenes import NeuralScene
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points
from pytorch3d.transforms import Transform3d, Translate, Rotate, Scale


class PointLightFieldComposition(nn.Module):
    def __init__(self,
                 scene: NeuralScene):

        super(PointLightFieldComposition, self).__init__()

        self._models = {}
        self._static_models = {}
        self._static_trafos = {}

        for node_types_dict in scene.nodes.values():
            for node in node_types_dict.values():
                if hasattr(node, 'lightfield'):
                    if not node.static:
                        self._models[node.scene_idx] = getattr(node, 'lightfield')
                    else:
                        self._static_models[node.scene_idx] = getattr(node, 'lightfield')
                        self._static_trafos[node.scene_idx] = node.transformation[:3, -1]

    def forward(self,
                ray_bundle: RayBundle,
                scene: NeuralScene,
                closest_point_mask,
                pt_cloud_select,
                closest_point_dist,
                closest_point_azimuth,
                closest_point_pitch,
                output_dict,
                ray_dirs_select,
                rotate2cam=False,
                **kwargs,
                ):
        device = ray_bundle.origins.device

        ray_bundle = ray_bundle._replace(
            lengths=ray_bundle.lengths.detach(),
            directions=ray_bundle.directions.detach(),
            origins=ray_bundle.origins.detach(),
        )

        # closest_point_mask, [pt_cloud_select, closest_point_dist, closest_point_azimuth, closest_point_pitch, output_dict], ray_dirs_select = pts

        xycfn = ray_bundle.xys

        c = torch.zeros_like(ray_bundle.origins)

        n_batch_rays = min([v.shape[0] for v in closest_point_dist.values()])
        # sample_mask = {
        #     cf_id :
        #         torch.stack(torch.where(torch.all(xycfn[..., 2:4] == torch.tensor(cf_id, device=device), dim=-1)))
        #     for cf_id in list(closest_point_dist.keys())
        # }
        sample_mask = {
            cf_id:
                torch.stack([
                    torch.ones(xycfn.shape[-2], dtype=torch.int64, device=device) * j,
                    torch.linspace(0, xycfn.shape[-2] - 1, xycfn.shape[-2], dtype=torch.int64, device=device)
                ])
            for j, cf_id in enumerate(list(closest_point_dist.keys()))
        }

        pt_cloud_select, closest_point_dist, closest_point_azimuth, closest_point_pitch, ray_dirs_select, closest_point_mask, sample_mask = \
        self.split_for_uneven_batch_sz(pt_cloud_select, closest_point_dist, closest_point_azimuth, closest_point_pitch, ray_dirs_select, closest_point_mask, sample_mask, n_batch_rays,
                                       device)

        for node_idx, model in self._static_models.items():
            x = None
            # TODO: Only get frame specific here
            for cf_id, mask in closest_point_mask.items():

                # Check if respective background and frame match
                frame = scene.frames[int(cf_id[1])]
                if any(node_idx == scene.frames[int(cf_id[1])].scene_matrix[:, 0]):
                    # TODO: Multiple frames at once
                    # Get projected rgb
                    # closest_rgb_fr = self._get_closest_rgb(scene, ray_bundle, point_cloud=pt_cloud_select[cf_id], c=int(cf_id[0]), f=int(cf_id[1]), device=device)
                    closest_rgb_fr = torch.zeros_like(pt_cloud_select[cf_id])

                    # TODO: rotation of the point cloud as part of the sampler or optional
                    if rotate2cam:
                        li_idx, li_node = list(scene.nodes['lidar'].items())[0]
                        assert li_node.name == "TOP"
                        cam_ed = frame.get_edge_by_child_idx([cf_id[0]])[0][0]
                        li_ed = frame.get_edge_by_child_idx([li_idx])[0][0]
                        # Transform x from li2world2cam
                        # li2world
                        li2wo = li_ed.get_transformation_c2p().to(device)
                        # world2cam
                        wo2cam = cam_ed.get_transformation_p2c().to(device)

                        pt_cloud_select[cf_id] = wo2cam.transform_points(li2wo.transform_points(pt_cloud_select[cf_id]))

                    # Check if intersections could be found inside frustum
                    if mask is not None:
                        if x is None:
                            x = pt_cloud_select[cf_id][None]
                            x_dist = closest_point_dist[cf_id][None]
                            azimuth = closest_point_azimuth[cf_id][None]
                            pitch = closest_point_pitch[cf_id][None]
                            ray_dirs = ray_dirs_select[cf_id][None]
                            closest_mask = closest_point_mask[cf_id][None]
                            # sample_mask = torch.stack(torch.where(torch.all(xycfn[..., 2:4] == cf_id, dim=-1)))[None]
                            closest_rgb = closest_rgb_fr[None]
                            projected_dist = ray_bundle.lengths[tuple(sample_mask[cf_id])][None]
                        else:
                            # print(cf_id)
                            # print(pt_cloud_select[cf_id][None].shape)
                            # print(x.shape)
                            x = torch.cat([x, pt_cloud_select[cf_id][None]])
                            x_dist = torch.cat([x_dist, closest_point_dist[cf_id][None]])
                            azimuth = torch.cat([azimuth, closest_point_azimuth[cf_id][None]])
                            pitch = torch.cat([pitch, closest_point_pitch[cf_id][None]])
                            ray_dirs = torch.cat([ray_dirs, ray_dirs_select[cf_id][None]])
                            closest_mask = torch.cat([closest_mask, closest_point_mask[cf_id][None]])
                            # new_fr_mask = torch.stack(torch.where(torch.all(xycfn[..., 2:4] == cf_id, dim=-1)))[None]
                            # sample_mask = torch.cat([sample_mask, new_fr_mask])
                            closest_rgb = torch.cat([closest_rgb, closest_rgb_fr[None]])

                            projected_dist = torch.cat([projected_dist, ray_bundle.lengths[tuple(sample_mask[cf_id])][None]])

            sample_idx = list(closest_point_mask.keys())
            if x is not None:
                if self.training:
                    # start = torch.cuda.Event(enable_timing=True)
                    # end = torch.cuda.Event(enable_timing=True)
                    # start.record()
                    color, output_dict = model(x, ray_dirs, closest_mask, x_dist, x_proj=projected_dist, x_pitch=pitch,
                                               x_azimuth=azimuth, rgb=closest_rgb, sample_idx=sample_idx)
                    # end.record()
                    # torch.cuda.synchronize()
                    # print('CUDA Time: {}'.format(start.elapsed_time(end)))
                else:
                    color, output_dict = model(x, ray_dirs, closest_mask, x_dist, x_proj=projected_dist, x_pitch=pitch,
                                               x_azimuth=azimuth, rgb=closest_rgb, sample_idx=sample_idx)
                for (sample_color, mask) in zip(color, list(sample_mask.values())):
                    c[tuple(mask)] = sample_color

        return c, output_dict


    def split_for_uneven_batch_sz(self, pt_cloud_select, closest_point_dist, closest_point_azimuth, closest_point_pitch, ray_dirs_select, closest_point_mask, sample_mask, n_batch_rays,
                                  device):
        for cf_id in list(closest_point_dist.keys()):
            v = closest_point_dist[cf_id]
            if v.shape[0] > n_batch_rays:
                factor = v.shape[0] // n_batch_rays
                for i in range(factor):
                    cf_id_new = cf_id + tuple([i * 1])
                    pt_cloud_select[cf_id_new] = pt_cloud_select[cf_id]
                    closest_point_dist[cf_id_new] = closest_point_dist[cf_id][i * n_batch_rays: (i + 1) * n_batch_rays]
                    closest_point_azimuth[cf_id_new] = closest_point_azimuth[cf_id][i * n_batch_rays: (i + 1) * n_batch_rays]
                    closest_point_pitch[cf_id_new] = closest_point_pitch[cf_id][i * n_batch_rays: (i + 1) * n_batch_rays]
                    ray_dirs_select[cf_id_new] = ray_dirs_select[cf_id][i * n_batch_rays: (i + 1) * n_batch_rays]
                    closest_point_mask[cf_id_new] = closest_point_mask[cf_id][i * n_batch_rays: (i + 1) * n_batch_rays]
                    sample_mask[cf_id_new] = sample_mask[cf_id][:, i * n_batch_rays: (i + 1) * n_batch_rays]

                del pt_cloud_select[cf_id]
                del closest_point_dist[cf_id]
                del closest_point_azimuth[cf_id]
                del closest_point_pitch[cf_id]
                del ray_dirs_select[cf_id]
                del closest_point_mask[cf_id]
                del sample_mask[cf_id]

        return pt_cloud_select, closest_point_dist, closest_point_azimuth, closest_point_pitch, ray_dirs_select, closest_point_mask, sample_mask


    def _get_closest_rgb(self, scene, ray_bundle, point_cloud, c, f, device):
        rgb = None
        img = torch.tensor(scene.frames[f].load_image(c), device=device)

        cam = scene.nodes['camera'][c]
        cam_ed = scene.frames[f].get_edge_by_child_idx([c])[0][0]
        # Get Camera Intrinsics and Extrensics
        cam_focal = cam.intrinsics.f_x
        cam_H = cam.intrinsics.H
        cam_W = cam.intrinsics.W
        cam_rot = Rotate(cam.R[None, ...].to(device), device=device)
        cam_transform = cam_ed.getTransformation()

        cam_w_xyz = point_cloud[:, [1, 2, 0]]
        cam_xy = cam_focal * (cam_w_xyz[:, [0, 1]] / cam_w_xyz[:, 2, None])
        cam_uv = -cam_xy + torch.tensor([[cam_W / 2, cam_H / 2]], device=device)
        cam_uv = cam_uv[:, [1, 0]].to(dtype=torch.int64)
        cam_uv = torch.maximum(cam_uv, torch.tensor(0, device=device))
        cam_uv[:, 0] = torch.minimum(cam_uv[:, 0], torch.tensor(cam_H - 1, device=device))
        cam_uv[:, 1] = torch.minimum(cam_uv[:, 1], torch.tensor(cam_W - 1, device=device))
        rgb = img[tuple(cam_uv.T)].detach()

        # img[tuple(cam_uv.T)] = np.array([1.0, 0., 0.])
        # f3 = plt.figure()
        # plt.imshow(img)

        return rgb