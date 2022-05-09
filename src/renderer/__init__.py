from typing import List, Dict
import copy
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import Counter
from ..scenes import NeuralScene
from pytorch3d.renderer.implicit.utils import RayBundle

from . import losses
from ..scenes import NeuralScene
from pytorch3d.renderer.implicit.utils import RayBundle

from ..pointLF.point_light_field import PointLightField
from ..pointLF.light_field_renderer import LightFieldRenderer
from ..pointLF.scene_point_lightfield import PointLightFieldComposition


class PointLightFieldRenderer(nn.Module):

    def __init__(self,
                 scene: NeuralScene,
                 chunk_size_test: int,
                 lightfield_config: Dict = {'k_closest': 30,
                                            'n_features': 8,
                                            'n_sample_pts': 5000},
                 point_chunk_size: int = 1e12,
                 device=None,
                 ):
        super(PointLightFieldRenderer, self).__init__()

        self.device = device

        self.scene = scene
        self._scene = scene
        self.lightfield_config = lightfield_config
        self._latent_codes = nn.ParameterDict()
        self._objs_size = nn.ParameterDict()
        self._cameras = nn.ParameterDict()
        self._poses = nn.ParameterDict()
        self._tone_maps = nn.ParameterDict()

        ## Add parameters
        self._init_trainable_scene_parameters(
            module='lightfield',
            train_obj_model=True,
            train_cam_pose=scene.refine_camera_pose,
            train_tone_mapping=scene.tonemapping,
        )

        # raysampler = PointLightFieldSampler(
        #     scene=self.scene,
        #     lightfield_config=lightfield_config,
        #     n_rays_per_image=n_rays_per_image,
        #     reference_frame='lidar',
        #     point_chunk_size=point_chunk_size
        # )

        # Pass sampling points through implicit functions
        self.light_field_module = PointLightFieldComposition(self.scene)

        self._density_noise_std = 0.
        self._latent_reg = 1e-7
        self._chunk_size_test = chunk_size_test
        self._transient_head = False

        # Add rendering function for forward pass here
        self.neural_renderer = LightFieldRenderer(light_field_module=self.light_field_module,
                                                  chunksize=self._chunk_size_test,
                                                  cam_centered=lightfield_config.get('camera_centered', False)
                                                  )

        # color = self._renderer.forward(scene=self._scene,
        #                                frame_idx=[0, 0],
        #                                camera_idx=[1, 0],
        #                                lightfield= self._scene_function)

    def _init_trainable_scene_parameters(self,
                                         module,
                                         train_obj_model,
                                         train_cam_pose=False,
                                         train_tone_mapping=False,
                                         ):
        # Add Camera poses to be refined
        if train_cam_pose:
            self._add_all_poses_delta_from_scene(cam_poses=True)

        if train_tone_mapping:
            self._add_all_tone_mapping_params()

        # Add object model Neural Point Light Fields
        for node_type, nodes in self._scene.nodes.items():
            for node_idx, node in nodes.items():

                # background node
                if node_type == 'background':
                    module_name = '_{}_{}_{}'.format(
                        node_type, module, str(node_idx).zfill(5)
                    )
                    lightfield = PointLightField(
                        k_closest=self.lightfield_config['k_closest'],
                        n_sample_pts=self.lightfield_config['n_sample_pts'],
                        n_pt_features=self.lightfield_config['n_features'],
                        feature_encoder=self.lightfield_config['pointfeat_encoder'],
                        feature_transform=True,
                        lf_architecture={
                            'D': self.lightfield_config.get('D_lf', 4),
                            'W': self.lightfield_config.get('W_lf', 256),
                            'skips': self.lightfield_config.get('skips_lf', []),
                            'modulation': self.lightfield_config.get('layer_modulation', False),
                            'poseEnc': self.lightfield_config.get('ray_encoding', 4),
                        },
                        new_encoding=self.lightfield_config.get('new_enc', False),
                        sky_dome=self.lightfield_config.get('sky_dome', False),
                    )

                    self.add_module(module_name, lightfield)
                    if train_obj_model is False:
                        for p in lightfield.parameters():
                            p.requires_grad = False

                    node.lightfield = lightfield

                # object nodes
                # TODO: add when necessary to model object lightfields

                # class nodes
                # TODO: add when necessary to model object lightfields

    def _add_all_tone_mapping_params(self):
        for f, fr in self.scene.frames.items():
            for c, wht_pt in fr.wht_pt.items():
                for color, val in wht_pt.items():
                    if val.requires_grad:
                        name = '{}_{}_{}_{}'.format(str(f).zfill(4), str(c).zfill(4), 'white', color)
                        parameter = self._add_parameter_from_scene(val, name, self._tone_maps)
                        fr.wht_pt[c][color] = parameter
                    else:
                        fr.wht_pt[c][color] = val.to(self.device)

                    val = fr.alpha_contrast[c]
                    if val.requires_grad:
                        name = '{}_{}_{}'.format(str(f).zfill(4), str(c).zfill(4), 'contrast', )
                        parameter = self._add_parameter_from_scene(val, name, self._tone_maps)
                        fr.alpha_contrast[c] = parameter

                    val = fr.beta_brightness[c]
                    if val.requires_grad:
                        name = '{}_{}_{}'.format(str(f).zfill(4), str(c).zfill(4), 'brightness', )
                        parameter = self._add_parameter_from_scene(val, name, self._tone_maps)
                        fr.beta_brightness[c] = parameter

                    val = fr.gamma[c]
                    if val.requires_grad:
                        name = '{}_{}_{}'.format(str(f).zfill(4), str(c).zfill(4), 'gamma', )
                        parameter = self._add_parameter_from_scene(val, name, self._tone_maps)
                        fr.gamma[c] = parameter


    def _add_all_poses_from_scene(self, cam_poses=False, obj_poses=False):
        # Get IDs of leaf nodes
        cam_scene_idx_ls = [i.scene_idx for i in self.scene.nodes['camera'].values()]
        scene_idx_obj = list(self.scene.nodes['scene_object'].keys())

        for frame_idx, frame in self.scene.frames.items():
            trainable_edges = []
            # Add Camera Transformations
            if cam_poses:
                trainable_edges += frame.get_edge_by_child_idx(cam_scene_idx_ls)

            # Add Object Transformations starting from leaf nodes
            if obj_poses:
                edges_all = frame.get_edge_by_child_idx(scene_idx_obj)
                trainable_edges += [edges for edges in edges_all if len(edges) != 0]

            for edge_list in trainable_edges:
                for edge in edge_list:
                    edge_name = 'frame_{}_edge_{}'.format(str(frame_idx).zfill(3), str(edge.index).zfill(3))
                    self._add_trainable_edge_attr(edge, 'translation', edge_name, self._poses)
                    self._add_trainable_edge_attr(edge, 'rotation', edge_name, self._poses)

    def _add_all_poses_delta_from_scene(self, cam_poses=False, obj_poses=False):
        # Get IDs of leaf nodes
        cam_scene_idx_ls = [i.scene_idx for i in self.scene.nodes['camera'].values()]
        scene_idx_obj = list(self.scene.nodes['scene_object'].keys())

        for frame_idx, frame in self.scene.frames.items():
            trainable_edges = []
            # Add Camera Transformations
            if cam_poses:
                trainable_edges += frame.get_edge_by_child_idx(cam_scene_idx_ls)

            # Add Object Transformations starting from leaf nodes
            if obj_poses:
                edges_all = frame.get_edge_by_child_idx(scene_idx_obj)
                trainable_edges += [edges for edges in edges_all if len(edges) != 0]

            for edge_list in trainable_edges:
                for edge in edge_list:
                    edge_name = 'frame_{}_edge_{}'.format(str(frame_idx).zfill(3), str(edge.index).zfill(3))
                    self._add_trainable_edge_attr(edge, 'delta_translation', edge_name, self._poses)
                    self._add_trainable_edge_attr(edge, 'delta_rotation', edge_name, self._poses)

    def _add_trainable_edge_attr(self, edge, attr_name, edge_name, dict):
        if hasattr(edge, attr_name):
            attr = getattr(edge, attr_name)
            parameter_name = '{}_{}'.format(edge_name, attr_name)
            parameter = self._add_parameter_from_scene(attr, parameter_name, dict)
            setattr(edge, attr_name, parameter)
        else:
            print('GraphEdge {} has no attribute \'{}\'.'.format(edge_name, attr_name))

    def _add_parameter_from_scene(self, parameter, name, dict):
        if not isinstance(parameter, torch.nn.Parameter):
            # try:
            parameter = nn.Parameter(parameter, requires_grad=True)
            dict[name] = parameter
            # print('Converting {} to a trainable paramter.'.format(str(name)))
            return parameter
        # except:
        #     print('Can not convert {} to \'torch.nn.Parameter\''.format(name))
        else:
            dict[name] = parameter
            return parameter

    def _process_ray_chunk(self,
                           frame_idx: List[int],
                           camera_idx: List[int],
                           chunk_idx: int,
                           bck_only,
                           obj_only,
                           obj_ids
                           ):
        density_noise_std = 0.
        output, ray_bundle_out = self._scene_renderer.forward(frame_idx=frame_idx,
                                                              camera_idx=camera_idx,
                                                              volumetric_function=self._scene_function,
                                                              chunk_idx=chunk_idx,
                                                              scene=self.scene, )

        if not self._transient_head:
            [rgb_map, output_dict] = output
        else:
            [rgb_map, output_dict] = output

        rgb = rgb_map

        rgb_gt = losses.get_rgb_gt(rgb, self.scene, xycfn=ray_bundle_out.xys)

        return {
            "rgb": rgb if self.training else rgb.detach().cpu(),
            "rgb_gt": rgb_gt if self.training else rgb_gt.detach().cpu(),
            # Store the rays/weights only for visualization purposes.
            "ray_bundle_out": type(ray_bundle_out)(
                *[v.detach().cpu() for k, v in ray_bundle_out._asdict().items()]
            ),
            "output_dict": output_dict,
        }


    def forward_on_batch(self, batch):

        ray_bundle, local_pts_model_idx = self.scene.raysampler.join_ray_bundle_batch(
            batch["ray_bundle"], None, device=self.device
        )

        # Extract object specific information for all sampling points
        # object_idx = [item for sublist in batch["object_idx"] for item in sublist]
        #
        # # TODO: make it work
        # new_config = isinstance(self.nerf_config.get('latent_size'), dict)
        # if new_config:
        #     latent_keys = list(self.nerf_config.get('latent_size', {'misc': None}).keys())
        #     object_latent = dict(zip(latent_keys,
        #                              [torch.stack([self.scene.nodes[int(n)]["node"].latent[k] for n in object_idx]) for
        #                               k in latent_keys]))
        # else:  # Old configuration:
        #     object_latent = {'misc': torch.stack(
        #         [self.scene.nodes[int(n)]["node"].latent for n in object_idx]
        #     )}

        # object_len = [item for sublist in batch["object_len"] for item in sublist]

        if self.training or self._chunk_size_test is None:
            # TRAINING
            input_dict = {
                "pts": batch['pts'],
                "ray_dirs_select": batch['pts_dirs'],
                "closest_point_mask": batch['pts_idx'],
                "ray_bundle": ray_bundle,
                # "object_idx": object_idx,
                # "object_latent": object_latent.to(self.device),
                # "object_len": object_len,
                # "object_input_latent": input_latent.to(self.device),
                # "image_latent": None,
                # "locs": None,
            }

            rgb_out = self.neural_renderer(
                input_dict=input_dict,
                scene=self.scene,
            )
            rgb_out.update({'xycfn': ray_bundle.xys.to(torch.int64)})

            gt_img = torch.stack(batch['images'])

            gt_img = torch.stack(
                [
                    gt_k[xy_k[:, 1], xy_k[:, 0]]
                    for xy_k, gt_k in zip(rgb_out['xycfn'][..., :2], gt_img)
                ]
            )
            rgb_out.update({'gt': gt_img.to(self.device)})

            return rgb_out
        else:
            ### EVALUATION

            # Use ray chunks during validation
            batch_sz, n_rays = ray_bundle.origins.shape[:-1]
            # For now validate on single images only
            assert batch_sz == 1

            input_dict = {
                "pts": batch['pts'],
                "ray_dirs_select": batch['pts_dirs'],
                "closest_point_mask": batch['pts_idx'],
                "ray_bundle": ray_bundle,
            }

            chunk_masks = self.chunk_input_dict(input_dict)

            rgb_out_chunk = []

            for k, mask in enumerate(chunk_masks):
                chunk_id = [0, k * self._chunk_size_test, 0]
                chunk_input_dict = copy.deepcopy(input_dict)

                chunk_input_dict.update(
                    {
                        "pts": [[input_dict['pts'][0][0]] +
                                [{list(dict.keys())[0]: list(dict.values())[0][mask]} for dict in input_dict['pts'][0][1:4]] +
                               [input_dict['pts'][0][4]]],
                        "ray_dirs_select": [{list(input_dict['ray_dirs_select'][0].keys())[0]:
                                                 list(input_dict['ray_dirs_select'][0].values())[0][mask]}],
                        "closest_point_mask": [{list(input_dict['closest_point_mask'][0].keys())[0]:
                                                    list(input_dict['closest_point_mask'][0].values())[0][mask]}],
                        "ray_bundle": RayBundle(
                            *[
                                v[
                                :,
                                k
                                * self._chunk_size_test: (k + 1)
                                                         * self._chunk_size_test,
                                ]
                                for v in input_dict["ray_bundle"]
                            ]
                        ),
                    }
                )

                out = self.neural_renderer(
                    input_dict=chunk_input_dict,
                    scene=self.scene,
                )
                rgb_out_chunk.append(out)

            rgb_out = {
                k: np.concatenate([outputs[k].squeeze() for j, outputs in enumerate(rgb_out_chunk)])
                for k in [
                    "points_selected_in",
                    "rays_in",
                    "closest_mask_in",
                    "sum_mv_point_features",
                    "attention_weights",
                    "selected_point_features",
                    "per_ray_features",
                    "color_out",
                ] if k in rgb_out_chunk[0].keys()}

            rgb_out.update({"rgb": torch.cat([outputs["rgb"].squeeze() for j, outputs in enumerate(rgb_out_chunk)]),
                            "ray_bundle": ray_bundle})

            rgb_out.update({k: rgb_out_chunk[0][k]
                            for k in [
                               "points_in",
                               "samples",
                               "points_scaled",
                               "raw_point_features"

                           ] if k in rgb_out_chunk[0].keys()})

            rgb_out.update({'xycfn': ray_bundle.xys.to(torch.int64)})

            gt_img = torch.stack(batch['images'])

            gt_img = torch.stack(
                [
                    gt_k[xy_k[:, 1], xy_k[:, 0]]
                    for xy_k, gt_k in zip(rgb_out['xycfn'][..., :2], gt_img)
                ]
            )
            rgb_out.update({'gt': gt_img.to(self.device)})

            return rgb_out

    def chunk_input_dict(self, input_dict):
        n_rays = input_dict["ray_bundle"].origins.shape[1]
        n_chunks = int(np.ceil(n_rays / self._chunk_size_test))
        chunk_start = np.arange(0, n_chunks) * self._chunk_size_test
        chunk_end = chunk_start + self._chunk_size_test

        chunk_masks = [np.linspace(start, end - 1, self._chunk_size_test, dtype=np.int64) for start, end in
         zip(chunk_start, chunk_end)]

        chunk_masks[-1] = chunk_masks[-1][chunk_masks[-1] < n_rays]

        return chunk_masks



    def compute_loss(self, rgb_out, gt, xycfn, weighted=False, **kwargs):
        y = rgb_out['gt']
        x = rgb_out['rgb'].reshape(y.shape)

        loss_dict = {}

        loss_dict["mse_loss"] = losses.mse_loss(x=x, y=y)

        return loss_dict

    def compute_psnr(self, pred, gt, rgb_out):
        y = rgb_out['gt']
        x = rgb_out['rgb'].reshape(y.shape)
        psnr = losses.calc_psnr(x=x, y=y)
        return psnr