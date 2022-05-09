from typing import Callable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from pytorch3d.renderer import RayBundle
from ..scenes import NeuralScene


class LightFieldRenderer(nn.Module):

    def __init__(self, light_field_module, chunksize: int, cam_centered: bool = False):
        super(LightFieldRenderer, self).__init__()

        self.light_field_module = light_field_module
        self.chunksize = chunksize
        if cam_centered:
            self.rotate2cam = True
        else:
            self.rotate2cam = False

    # def forward(self, frame_idx: int, camera_idx: int, scene: NeuralScene, volumetric_function: Callable, chunk_idx: int, **kwargs):
    def forward(self, input_dict, scene, **kwargs):

        ##############
        # TODO: Clean Up and move somewhere else
        ray_bundle = input_dict['ray_bundle']
        device = ray_bundle.origins.device
        pts = input_dict['pts']
        ray_dirs_select = input_dict['ray_dirs_select']
        closest_point_mask = input_dict['closest_point_mask']

        cf_ls = [list(pts_k[0].keys()) for pts_k in pts]
        import numpy as np
        unique_cf = np.unique(np.concatenate([np.array(cf) for cf in cf_ls]), axis=0)

        pts_to_unpack = {
            0: 'pt_cloud_select',
            1: 'closest_point_dist',
            2: 'closest_point_azimuth',
            3: 'closest_point_pitch',

        }

        if len(unique_cf) != len(cf_ls):
            new_cf = [tuple(list(cf[0]) + [j]) for j, cf in enumerate(cf_ls)]

            output_dict = {
                new_cf[j]:
                    v
                for j, pt in enumerate(pts) for k, v in pt[4].items()
            }

            closest_point_mask = {new_cf[j]: v for j, pt in enumerate(closest_point_mask) for k, v in pt.items()}
            ray_dirs_select = {new_cf[j]: v.to(device) for j, pt in enumerate(ray_dirs_select) for k, v in pt.items()}
            pts = {
                n: {
                    new_cf[j]:
                        v.to(device)
                    for j, pt in enumerate(pts) for k, v in pt[i].items()
                }
                for i, n in pts_to_unpack.items()
            }
            pts.update({'output_dict': output_dict})

            a = 0

        else:
            output_dict = {
                k:
                    v
                for pt in pts for k, v in pt[4].items()
            }

            closest_point_mask = {k: v for pt in closest_point_mask for k, v in pt.items()}
            ray_dirs_select = {k: v.to(device) for pt in ray_dirs_select for k, v in pt.items()}
            pts = {
                n: {
                    k:
                        v.to(device)
                    for pt in pts for k, v in pt[i].items()
                    }
                for i, n in pts_to_unpack.items()
            }
            pts.update({'output_dict': output_dict})
        ##################

        images, output_dict = self.light_field_module(
            ray_bundle=input_dict['ray_bundle'],
            scene=scene,
            closest_point_mask=closest_point_mask,
            pt_cloud_select=pts['pt_cloud_select'],
            closest_point_dist=pts['closest_point_dist'],
            closest_point_azimuth=pts['closest_point_azimuth'],
            closest_point_pitch=pts['closest_point_pitch'],
            output_dict=pts['output_dict'],
            ray_dirs_select=ray_dirs_select,
            rotate2cam=self.rotate2cam,
            **kwargs
        )

        if scene.tonemapping:
            rgb = torch.zeros_like(images)
            tone_mapping_ls = [scene.frames[cf[0][1]].load_tone_mapping(cf[0][0]) for cf in cf_ls]
            for i in range(len(images)):
                rgb[i] = self.tone_map(images[i], tone_mapping_ls[i])

        else:
            rgb = images

        output_dict.update(
            {
                'rgb': rgb.view(-1, 3),
                'ray_bundle': ray_bundle._replace(xys=ray_bundle.xys[..., None, :])
            }
        )

        return output_dict

    def tone_map(self, x, tone_mapping_params):

        x = (tone_mapping_params['contrast'] * (x - 0.5) + \
             0.5 + \
             tone_mapping_params['brightness']) * \
            torch.cat(list(tone_mapping_params['wht_pt'].values()))
        x = self.leaky_clamping(x, gamma=tone_mapping_params['gamma'])

        return x

    def leaky_clamping(self, x, gamma, alpha=0.01):
        x[x < 0] = x[x < 0] * alpha
        x[x > 1] = (-alpha / x[x > 1]) + alpha + 1.
        x[(x > 0.) & (x < 1.)] = x[(x > 0.) & (x < 1.)] ** gamma
        return x