import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointcloud_encoding.pointnet_features import PointNetDenseCls, PointNetLightFieldEncoder
from .pointcloud_encoding.simpleview import MVModel
from src.pointLF.attention_modules import PointFeatureAttention, PointDistanceAttention
from src.pointLF.layer import *
from src.pointLF.pointLF_helper import *
from src.pointLF.ptlf_vis import *
from src.pointLF.feature_mapping import PositionalEncoding

# Takes Points, weights  and rays and maps to color
class PointLightField(nn.Module):
    def __init__(self, k_closest=30, n_sample_pts=1000, n_pt_features=8, feature_encoder='pointnet_segmentation' ,feature_transform=True,
                 lf_architecture={'D': 4,
                                  'W': 256,
                                  'skips': [],
                                  'modulation': False,
                                  'poseEnc': 4,
                                  },
                 new_encoding=False,
                 sky_dome=False):
        super(PointLightField, self).__init__()

        self.new_enc = new_encoding

        self.feat_weighting = None
        self._RGBFeatures = False
        self.n_pt_features = n_pt_features
        self.pre_scale = False
        self.no_feat = False
        self.stored_feature_maps = {}
        self.stored_points_in = None
        layer_modulation = False
        n_feat_in = k_closest
        upscale_feat_maps=False

        if feature_encoder == 'pointnet_segmentation':
            self._PointFeatures = PointNetDenseCls(k=n_pt_features, feature_transform=feature_transform)
        elif feature_encoder == 'pointnet_lf_pt_only':
            self._PointFeatures = PointNetLightFieldEncoder(k=n_pt_features, feature_transform=feature_transform,
                                                            points_only=True)
        elif feature_encoder == 'pointnet_lf_global':
            self._PointFeatures = PointNetLightFieldEncoder(k=n_pt_features, feature_transform=feature_transform,
                                                            points_only=False)
        elif feature_encoder == 'pointnet_lf_pt_only_weighted':
            self._PointFeatures = PointNetLightFieldEncoder(k=n_pt_features, feature_transform=feature_transform,
                                                            points_only=True)
            self.feat_weighting = 'lin'
        elif feature_encoder == 'pointnet_lf_global_weighted':
            self._PointFeatures = PointNetLightFieldEncoder(k=n_pt_features, feature_transform=feature_transform,
                                                            points_only=False)
            self.feat_weighting = 'lin'
        elif feature_encoder == 'projected_colour':
            self._PointFeatures = lambda x, rgb : [rgb, None, None]
            self._RGBFeatures = True
            self.n_pt_features = 3
        elif feature_encoder == 'multiview' or feature_encoder == 'multiview_encoded':
            self._PointFeatures = MVModel(task='cls', backbone='resnet18', feat_size=16)
            self.n_pt_features = 128
            self.pre_scale = True
            self.feat_weighting = 'max_pool'
        elif feature_encoder == 'multiview_attention_modulation':
            self._PointFeatures = MVModel(task='cls', backbone='resnet18', feat_size=16)
            self.n_pt_features = 128
            self.key_len = 64
            self.pre_scale = True
            self.feat_weighting = 'attention'
            self.AttentionModule = PointFeatureAttention(feat_dim_in=self.n_pt_features,
                                                         feat_dim_out=self.n_pt_features,
                                                         embeded_dim=256, n_att_heads=8,
                                                         kdim = 128, vdim = 128, )
            n_feat_in = 1
            lf_architecture.update({'modulation': True})
        elif feature_encoder == 'multiview_attention':
            self._PointFeatures = MVModel(task='cls', backbone='resnet18', feat_size=16)
            self.n_pt_features = 128
            self.key_len = 64
            self.pre_scale = True
            self.feat_weighting = 'attention'
            self.AttentionModule = PointFeatureAttention(feat_dim_in=self.n_pt_features,
                                                         feat_dim_out=self.n_pt_features,
                                                         embeded_dim=256, n_att_heads=8,
                                                         kdim=128, vdim=128,
                                                         new_encoding=self.new_enc)
            n_feat_in = 1

        elif feature_encoder == 'heuristic_ablation':
            self._PointFeatures = MVModel(task='cls', backbone='resnet18', feat_size=16)
            self.n_pt_features = 128
            self.key_len = 64
            self.pre_scale = True
            self.feat_weighting = 'lin'
            self.AttentionModule = PointFeatureAttention(feat_dim_in=self.n_pt_features,
                                                         feat_dim_out=self.n_pt_features,
                                                         embeded_dim=256, n_att_heads=8,
                                                         kdim=128, vdim=128,
                                                         new_encoding=self.new_enc)
            n_feat_in = 1
        elif feature_encoder == 'naive_ablation':
            self._PointFeatures = MVModel(task='cls', backbone='resnet18', feat_size=16)
            self.n_pt_features = 128
            self.key_len = 64
            self.pre_scale = True
            self.feat_weighting = 'sum'
            self.AttentionModule = PointFeatureAttention(feat_dim_in=self.n_pt_features,
                                                         feat_dim_out=self.n_pt_features,
                                                         embeded_dim=256, n_att_heads=8,
                                                         kdim=128, vdim=128,
                                                         new_encoding=self.new_enc)
            n_feat_in = 1
        elif feature_encoder == 'one_point_ablation':
            self._PointFeatures = MVModel(task='cls', backbone='resnet18', feat_size=16)
            self.n_pt_features = 128
            self.key_len = 64
            self.pre_scale = True
            self.feat_weighting = 'single_choice'
            self.AttentionModule = PointFeatureAttention(feat_dim_in=self.n_pt_features,
                                                         feat_dim_out=self.n_pt_features,
                                                         embeded_dim=256, n_att_heads=8,
                                                         kdim=128, vdim=128,
                                                         new_encoding=self.new_enc)
            n_feat_in = 1
        elif feature_encoder == 'pointnet_ablation':
            self._PointFeatures = PointNetLightFieldEncoder(k=self.n_pt_features, feature_transform=feature_transform,
                                                            points_only=False, original=True)
            self.n_pt_features = 128
            self.key_len = 64
            self.pre_scale = False
            self.feat_weighting = 'attention'
            self.AttentionModule = PointFeatureAttention(feat_dim_in=self.n_pt_features,
                                                         feat_dim_out=self.n_pt_features,
                                                         embeded_dim=256, n_att_heads=8,
                                                         kdim=128, vdim=128,
                                                         new_encoding=self.new_enc)
            n_feat_in = 1

        elif feature_encoder == 'pointnet_local_only_ablation':
            self._PointFeatures = PointNetLightFieldEncoder(k=self.n_pt_features, feature_transform=feature_transform,
                                                            points_only=True, original=False)
            self.n_pt_features = 128
            self.key_len = 64
            self.pre_scale = False
            self.feat_weighting = 'attention'
            self.AttentionModule = PointFeatureAttention(feat_dim_in=self.n_pt_features,
                                                         feat_dim_out=self.n_pt_features,
                                                         embeded_dim=256, n_att_heads=8,
                                                         kdim=128, vdim=128,
                                                         new_encoding=self.new_enc)
            n_feat_in = 1

        elif feature_encoder == 'encoding_attention_only':
            self._PointFeatures = lambda x, rgb : (x, None, None)
            self.n_pt_features = 128
            self.key_len = 64
            self.pre_scale = False
            self.feat_weighting = 'attention'
            self.no_feat = True
            self.AttentionModule = PointFeatureAttention(feat_dim_in=self.n_pt_features,
                                                         feat_dim_out=self.n_pt_features,
                                                         embeded_dim=256, n_att_heads=8,
                                                         kdim=128, vdim=128,
                                                         new_encoding=self.new_enc,
                                                         no_feat=True)
            n_feat_in = 1
        elif feature_encoder == 'multiview_attention_up':
            self._PointFeatures = MVModel(task='cls', backbone='resnet18', feat_size=16, resolution=128, upscale_feats=True)
            self.n_pt_features = 128
            self.key_len = 64
            self.pre_scale = True
            self.feat_weighting = 'attention'
            self.AttentionModule = PointFeatureAttention(feat_dim_in=self.n_pt_features,
                                                         feat_dim_out=self.n_pt_features,
                                                         embeded_dim=256, n_att_heads=8,
                                                         kdim=128, vdim=128,
                                                         new_encoding=self.new_enc)
            n_feat_in = 1
        elif feature_encoder == 'multiview_attention_big':
            # DIFFERENCE: Resolution of projected point cloud is 512
            self._PointFeatures = MVModel(task='cls', backbone='resnet18', feat_size=16, resolution=512)
            self.n_pt_features = 128
            self.key_len = 64
            self.pre_scale = True
            self.feat_weighting = 'attention'
            self.AttentionModule = PointFeatureAttention(feat_dim_in=self.n_pt_features,
                                                         feat_dim_out=self.n_pt_features,
                                                         embeded_dim=256, n_att_heads=8,
                                                         kdim=128, vdim=128, )
            n_feat_in = 1
        # elif feature_encoder == 'multiview_distance_attention':
        #     self._PointFeatures = MVModel(task='cls', backbone='resnet18', feat_size=16)
        #     self.n_pt_features = 128
        #     self.key_len = 16
        #     self.pre_scale = True
        #     self.feat_weighting = 'attention'
        #     self.AttentionModule = PointDistanceAttention(v_len=self.n_pt_features, kq_len=self.key_len)
        #     n_feat_in = 1
        #     layer_modulation = False
        #     ray_encoding = True
        elif feature_encoder == 'multiview_encoded_modulation':
            self._PointFeatures = MVModel(task='cls', backbone='resnet18', feat_size=16)
            self.n_pt_features = 128 * k_closest
            self.pre_scale = True
            self.feat_weighting = 'max_pool'
            lf_architecture.update({'modulation': True})
        elif feature_encoder == 'multiview_encoded_weighted_modulation':
            self._PointFeatures = MVModel(task='cls', backbone='resnet18', feat_size=16)
            self.n_pt_features = 128
            self.pre_scale = True
            self.feat_weighting = 'lin'
            lf_architecture.update({'modulation': True})
        else:
            ValueError('{} feature encoder does not exist.'.format(feature_encoder))



        self._LightField = LightFieldNet(n_feat_in=n_feat_in,
                                         n_pt_feat=self.n_pt_features,
                                         D=lf_architecture['D'],
                                         W=lf_architecture['W'],
                                         multires=lf_architecture['poseEnc'],
                                         skips=lf_architecture['skips'],
                                         layer_modulation=lf_architecture['modulation'])

        self.sky_dome = sky_dome
        if self.sky_dome:
            self._sky_latent = nn.Parameter(torch.rand(self.n_pt_features), requires_grad=True)
            self._SkyField = LightFieldNet(n_feat_in=n_feat_in,
                                           n_pt_feat=self.n_pt_features,
                                           D=lf_architecture['D']//2,
                                           W=lf_architecture['W'],
                                           multires=lf_architecture['poseEnc'],
                                           skips=[],
                                           layer_modulation=False)


        self._k_closest = k_closest
        self._n_pts = n_sample_pts


    def forward(self, x, ray_dirs=None, closest_mask=None, x_dist=None, x_proj=None, x_pitch=None, x_azimuth=None, rgb=None,
                sample_idx=None):
        """
        x: Points [batch_size, N_pts, n_feat]
        x_dist: [N_rays, N_pts, 1]
        ray_bundle: [1, N_rays, 3]
        """
        self.first_val_pass = True
        if not self.training and self.stored_points_in is not None:
            dist = np.abs(np.sum(self.stored_points_in - x.detach().cpu().numpy()))
            if dist < 1e-5:
                self.first_val_pass = False
            else:
                self.stored_points_in = x.detach().cpu().numpy().copy()
        elif not self.training:
            self.stored_points_in = x.detach().cpu().numpy().copy()



        if self.training or self.first_val_pass:
            output_dict = {'points_in': x.detach().cpu().numpy().copy(),
                           'points_selected_in': np.stack(
                               [pts[mask].detach().cpu().numpy() for (pts, mask) in zip(x, closest_mask)]
                           ),
                           'rays_in': ray_dirs.detach().cpu().numpy().copy(),
                           'closest_mask_in': closest_mask.detach().cpu().numpy().copy(),
                           'samples': np.stack([np.array(smpl_id)[:2] for smpl_id in sample_idx])[:, None]}
        else:
            output_dict = {'points_selected_in': np.stack(
                               [pts[mask].detach().cpu().numpy() for (pts, mask) in zip(x, closest_mask)]
                           ),
                           'rays_in': ray_dirs.detach().cpu().numpy().copy(),
                           'closest_mask_in': closest_mask.detach().cpu().numpy().copy(),
                           'samples': np.stack([np.array(smpl_id)[:2] for smpl_id in sample_idx])[:, None]}


        batchsize = x.size()[0]
        n_batchrays = closest_mask.size()[1]

        # c = torch.zeros([batchsize, n_batchrays, 3], device=x.device)

        if self._k_closest:

            # 1. Extract PointNet features [N, n_feat, 1]
            if self.pre_scale:
                # Transform points inside a 1 by 1 sized cube
                pts_x = pre_scale_MV(x)
                if self.training or self.first_val_pass:
                    output_dict['points_scaled'] = pts_x.detach().cpu().numpy().copy()

            else:
                pts_x = x[..., :3].transpose(2, 1)

            feat, trans, trans_feat = self._PointFeatures(pts_x, rgb=rgb)

            # TODO: Get feature back also in other parts of the pipeline
            # if self.training:
            #     feat, trans, trans_feat = self._PointFeatures(pts_x, rgb=rgb)
            #     self.stored_feature_maps = {}
            # else:
            #     # During validation only calculate point cloud features, when new pointclouds are evaluated
            #     for sample in sample_idx:
            #         if not sample in self.stored_feature_maps:
            #             feat, trans, trans_feat = self._PointFeatures(pts_x, rgb=rgb)
            #             for sample, sample_feat in zip(sample_idx, feat):
            #                 self.stored_feature_maps[sample] = sample_feat.detach().cpu()
            #             break
            #
            #     feat = torch.stack([self.stored_feature_maps[sample].to(pts_x.device) for sample in sample_idx])

            feat_raw = feat.detach().cpu().numpy().copy()
            # 2. Select the relevant features
            if not self.no_feat:
                if self.pre_scale:
                    # Re-project features from features maps at a cubes phases to the closest points of a ray
                    feat = select_Mv_feat(feat, pts_x, closest_mask, batchsize,
                                          k_closest=self._k_closest, feature_extractor=self._PointFeatures,
                                          img_resolution=trans.shape[-1], feature_resolution=feat.shape[-1])
                else:
                    # Select closest features to the ray from the precalculated mask
                    feat = [pts_feat[mask] for (pts_feat, mask) in zip(feat, closest_mask)]
                    feat = torch.stack(feat)

                feat_sel_proj = feat.detach().cpu().numpy().copy()

            # 3. Apply weighting on point features...
            if self.feat_weighting == 'lin':
                n_feat_per_point = feat.shape[-2]
                feat = torch.sum(feat, dim=-2)[..., None, :] / n_feat_per_point
                # ...given distance to the ray

                inv_dist = torch.div(1, x_dist)
                dist_weight = inv_dist / torch.sum(inv_dist, dim=-1)[..., None]
                feat = torch.sum(feat.squeeze() * dist_weight[..., None], dim=2)
                # feat = lin_weighting(z=feat, distance=x_dist, projected=x_proj)
                # feat = feat.mean(dim=2)

            elif self.feat_weighting == 'attention' and self.no_feat:
                feat = torch.zeros(batchsize, n_batchrays, self._k_closest, 1, self.n_pt_features)
                feat_sel_proj = feat.detach().cpu().numpy().copy()
                feat, attn_weights = self.AttentionModule(directions=ray_dirs, features=feat, distance=x_dist,
                                                          projected_distance=x_proj, pitch=x_pitch, azimuth=x_azimuth,)
                output_dict.update(
                    {
                        'attention_weights':
                            attn_weights.view(batchsize, n_batchrays, -1, self._k_closest).cpu().detach().numpy()
                    }
                )

            elif self.feat_weighting == 'attention':
                # ...by a learned attention module
                if feat.dim() == 5:
                    n_feat_per_point = feat.shape[-2]
                    # feat = torch.sum(feat, dim=-2)[..., None, :]
                    # feat = torch.max(feat, dim=-2)[0][..., None, :]
                    feat = torch.sum(feat, dim=-2)[..., None, :] / n_feat_per_point

                    output_dict.update(
                        {
                            'sum_mv_point_features': feat.detach().cpu().numpy().copy(),
                        }
                    )
                else:
                    n_feat_per_point = 1
                    feat = feat[..., None, :]

                # Add global features for rays, that don't hit areas represented by the the point cloud
                if self.sky_dome:
                    sky_threshold = .4
                    sky_mask = torch.where(x_dist.min(dim=-1)[0] > sky_threshold)
                    point_mask = torch.where(x_dist.min(dim=-1)[0] <= sky_threshold)
                    n_sky_rays = len(sky_mask[0])


                if not self.new_enc and not self.sky_dome:
                    feat, attn_weights = self.AttentionModule(ray_dirs, feat, x_proj, x_dist, pitch=x_pitch, azimuth=x_azimuth)
                    output_dict.update(
                        {
                            'attention_weights':
                                attn_weights.view(batchsize, n_batchrays, -1, self._k_closest).cpu().detach().numpy()
                        }
                    )

                elif self.sky_dome:
                    if len(sky_mask[0]) > 0:
                        sky_rays, sky_feat, sky_dist, sky_proj, sky_pitch, sky_azimuth = \
                            self.add_sky_features(n_sky_rays, sky_mask, ray_dirs, feat, x_dist, x_proj, x_pitch, x_azimuth, device=x.device)

                        sky_feat, sky_attn_weights = self.AttentionModule(directions=sky_rays,
                                                                          features=sky_feat,
                                                                          distance=sky_dist,
                                                                          projected_distance=sky_proj,
                                                                          pitch=sky_pitch,
                                                                          azimuth=sky_azimuth
                                                                          )


                    if len(point_mask[0]) > 0:
                        point_feat, point_attn_weights = self.AttentionModule(directions=ray_dirs[point_mask][None],
                                                                        features=feat[point_mask][None],
                                                                        distance=x_dist[point_mask][None],
                                                                        projected_distance=x_proj[point_mask][None],
                                                                        pitch=x_pitch[point_mask][None],
                                                                        azimuth=x_azimuth[point_mask][None],
                                                                        )


                    feat = torch.zeros([batchsize, n_batchrays, self.n_pt_features], device=x.device)
                    # Store attention weights for debugging/visualization
                    attn_weights = torch.zeros([batchsize, n_batchrays, self._k_closest + 1], device=x.device)

                    if len(sky_mask[0]) > 0:
                        feat[sky_mask] += sky_feat.squeeze()
                        attn_weights[sky_mask] += sky_attn_weights.squeeze()

                    if len(point_mask[0]) > 0:
                        feat[point_mask] += point_feat.squeeze()
                        point_attn_weights = torch.cat(
                            [point_attn_weights, torch.zeros([len(point_mask[0]), 1, 1], device=x.device)], dim=-1)
                        attn_weights[point_mask] += point_attn_weights.squeeze()

                    output_dict.update(
                        {
                            'attention_weights':
                                attn_weights.view(batchsize, n_batchrays, -1, self._k_closest+1).cpu().detach().numpy()
                        }
                    )

                else:
                    feat, attn_weights = self.AttentionModule(directions=ray_dirs, features=feat, distance=x_dist,
                                                              projected_distance=x_proj, pitch=x_pitch, azimuth=x_azimuth)

                    output_dict.update(
                        {
                        'attention_weights':
                            attn_weights.view(batchsize, n_batchrays, -1, self._k_closest).cpu().detach().numpy()
                        }
                    )

            elif self.feat_weighting == 'max_pool':
                feat = torch.max(feat, dim=-2)[0]

            elif self.feat_weighting =='sum':
                n_feat_per_point = feat.shape[-2]
                feat = torch.sum(feat, dim=-2)[..., None, :] / n_feat_per_point
                feat = torch.sum(feat[..., 0, :], dim=-2)
            elif self.feat_weighting == 'single_choice':
                n_feat_per_point = feat.shape[-2]
                single_idx = tuple(
                    [torch.linspace(0, batchsize - 1, batchsize, dtype=torch.int64)[:, None].repeat(1,n_batchrays),
                     torch.linspace(0, n_batchrays - 1, n_batchrays, dtype=torch.int64)[None, :].repeat(batchsize, 1),
                     torch.argsort(x_dist)[..., 0]
                     ]
                )
                feat = feat[single_idx]
                feat = torch.sum(feat, dim=-2)
                feat, attn_weights = self.AttentionModule(
                    directions=ray_dirs,
                    features=feat[..., None, None, :],
                    distance=x_dist[single_idx][..., None],
                    projected_distance=x_proj[single_idx][..., None],
                    pitch=x_pitch[single_idx][..., None],
                    azimuth=x_azimuth[single_idx][..., None],
                )


            output_dict.update(
                {
                    'raw_point_features': feat_raw,
                    'selected_point_features': feat_sel_proj,
                    'per_ray_features': feat.detach().cpu().numpy().copy(),
                }
            )
        else:
            feat = torch.zeros(list(ray_dirs.shape[:2]) + [128], dtype=x.dtype, device=x.device)

        # 4. Use PointNet Features and location Inputs to predict light field outputs
        c = self._LightField(ray_dirs, feat)

        # if self.sky_dome:
        #     c_sky = self._SkyField(
        #         ray_dirs[sky_mask].unsqueeze(0),
        #         feat[sky_mask].unsqueeze(0)
        #     )
        #
        #     c[sky_mask] *= 0.
        #     c[sky_mask] += c_sky.squeeze()

        output_dict.update(
            {
                'color_out': c.detach().cpu().numpy(),
            }
        )

        # 5. Visualization
        # visualize_output(output_dict, scaled=False, selected_only=True, n_plt_rays=None)
        # plt_BEV_pts_selected(output_dict)
        # plt_SIDE_pts_selected(output_dict)
        # plt_FRONT_pts_selected(output_dict)

        return c, output_dict


    def add_sky_features(self, n_sky_rays, sky_mask, ray_dirs, feat, x_dist, x_proj, x_pitch, x_azimuth, device):
        sky_rays = ray_dirs[sky_mask].unsqueeze(0)
        sky_feat = feat[sky_mask].unsqueeze(0)
        sky_feat = torch.cat([sky_feat,
                              self._sky_latent.expand(
                                  list(sky_feat.shape[:2]) + [1] + list(sky_feat.shape[3:])
                              )], dim=2)
        sky_dist = torch.cat([x_dist[sky_mask].unsqueeze(0), torch.zeros([1, n_sky_rays, 1], device=device)], dim=-1)
        sky_proj = torch.cat([x_proj[sky_mask].unsqueeze(0), torch.zeros([1, n_sky_rays, 1], device=device)], dim=-1)
        sky_pitch = torch.cat([x_pitch[sky_mask].unsqueeze(0), torch.zeros([1, n_sky_rays, 1], device=device)],
                              dim=-1)
        sky_azimuth = torch.cat([x_azimuth[sky_mask].unsqueeze(0), torch.zeros([1, n_sky_rays, 1], device=device)],
                                dim=-1)

        return sky_rays, sky_feat, sky_dist, sky_proj, sky_pitch, sky_azimuth


class LightFieldNet(nn.Module):
    def __init__(self,
                 n_feat_in,
                 n_pt_feat,
                 layer_modulation=False,
                 D=4,
                 W=256,
                 multires=6,
                 skips=[]):
        super(LightFieldNet, self).__init__()

        self.D = D
        self.W = W
        self.n_feat_in = n_feat_in
        self.feat_ch = n_pt_feat

        self.pose_encoding = PositionalEncoding(multires)


        # self.input_ch = k * (n_pt_feat + 3)

        self.skips = skips

        if not layer_modulation:
            Layer = DenseLayer
            self.input_ch = self.n_feat_in * (n_pt_feat + (multires * 2 * 3 + 3))
        else:
            Layer = ModulationLayer
            self.input_ch = (multires * 2 * 3 + 3)

        self.layer_modulation = layer_modulation

        self.pts_linears = nn.ModuleList(
            [Layer(self.input_ch, W, z_dim=n_pt_feat)] +
            [Layer(W, W, z_dim=n_pt_feat)
             if i not in self.skips else
             Layer(W + self.input_ch, W, z_dim=n_pt_feat)
             for i in range(D - 1)]
        )

        self.rgb_linear = DenseLayer(W, 3, activate=False)


    def forward(self, ray_dirs, z):
        """
        ray_dirs: [batch_size, N_rays, 3]
        z: Latent encoding of the Light Field [batch_size, N_rays, k, n_feat]
        """
        if z.dim() == 4:
            n_batch, n_rays, n_feat, feat_ch = z.shape
            assert n_feat == self.n_feat_in
            if not self.layer_modulation:
                ray_dirs = ray_dirs[..., None, :].repeat(1, 1, n_feat, 1)
        else:
            n_batch, n_rays, feat_ch = z.shape
            n_feat = 1

        ray_dirs = self.pose_encoding(ray_dirs)

        if not self.layer_modulation:
            inputs = torch.cat([z, ray_dirs], dim=-1).view(n_batch, -1, self.input_ch)
            h = inputs
        else:
            h = ray_dirs.view(n_batch* n_rays, self.input_ch)
            z = z.view(n_batch * n_rays, self.feat_ch)

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h, z=z)
            if i in self.skips:
                h = torch.cat([inputs, h], -1)

        out = self.rgb_linear(h, z=z)
        rgb = torch.sigmoid(out)

        return rgb.view(n_batch, n_rays, 3)