import torch
import torch.nn as nn
import torch.nn.functional as F

def pre_scale_MV(x, translation=-1.4):
    batchsize = x.size()[0]
    scaled_x = x

    for k in range(batchsize):
        for i in range(3):
            max = scaled_x[k, ..., i].max()
            min = scaled_x[k, ..., i].sort()[0][10]
            ax_size = max - min
            scaled_x[k, ..., i] -= min
            scaled_x[k, ..., i] *= 2 / ax_size
            scaled_x[k, ..., i] -= 1.

    scaled_x = torch.minimum(scaled_x, torch.tensor([1.], device='cuda'))
    scaled_x = torch.maximum(scaled_x, torch.tensor([-1.], device='cuda'))

    # scaled_x[:, 0] = torch.tensor([-1.0, -1.0, -1.0], device=scaled_x.device)
    # scaled_x[:, 1] = torch.tensor([1.0, -1.0, -1.0], device=scaled_x.device)
    # scaled_x[:, 2] = torch.tensor([-1.0, 1.0, -1.0], device=scaled_x.device)
    # scaled_x[:, 3] = torch.tensor([1.0, 1.0, -1.0], device=scaled_x.device)
    # scaled_x[:, 4] = torch.tensor([-1.0, -1.0, 1.0], device=scaled_x.device)
    # scaled_x[:, 5] = torch.tensor([-1.0, 1.0, 1.0], device=scaled_x.device)
    # scaled_x[:, 6] = torch.tensor([1.0, -1.0, 1.0], device=scaled_x.device)
    # scaled_x[:, 7] = torch.tensor([1.0, 1.0, 1.0], device=scaled_x.device)

    scaled_x *= 1 / -translation

    # scaled_plt = scaled_x[0].cpu().detach().numpy()
    # fig3d = plt.figure()
    # ax3d = fig3d.gca(projection='3d')
    # ax3d.scatter(scaled_plt[:, 0], scaled_plt[:, 1], scaled_plt[:, 2], c='blue')

    return scaled_x


def select_Mv_feat(feature_maps, scaled_pts, closest_mask, batchsize, k_closest, feature_extractor, img_resolution=128,
                   feature_resolution=16):
    feat2img_f = img_resolution // feature_resolution

    # n_feat_maps, batchsize, n_features, feat_heigth, feat_width  = feature_maps.shape
    n_batch, maps_per_batch, n_features, feat_heigth, feat_width = feature_maps.shape
    n_feat_maps = maps_per_batch * n_batch
    feature_maps = feature_maps.reshape(n_feat_maps, n_features, feat_heigth, feat_width)

    # Only retrive pts_feat for relevant points
    masked_scaled_pts = [sc_x[mask] for (sc_x, mask) in zip(scaled_pts, closest_mask)]
    masked_scaled_pts = torch.stack(masked_scaled_pts).view(n_batch, -1, 3)

    # Get coordinates in the feautre maps for each point
    coordinates, coord_x, coord_y, depth = feature_extractor._get_img_coord(masked_scaled_pts, resolution=img_resolution)

    # Adjust for downscaled feature maps
    coord_x = torch.round(coord_x.view(n_feat_maps, -1, k_closest) / feat2img_f).to(torch.long)
    coord_x = torch.minimum(coord_x, torch.tensor([feat_heigth - 1], device=coord_x.device))
    coord_y = torch.round(coord_y.view(n_feat_maps, -1, k_closest) / feat2img_f).to(torch.long)
    coord_y = torch.minimum(coord_y, torch.tensor([feat_width - 1], device=coord_x.device))

    # depth = depth.view(n_batch, maps_per_batch, -1, k_closest)

    # Extract features for each ray and k closest points
    feature_maps = feature_maps.permute(0, 2, 3, 1)
    pts_feat = torch.stack([feature_maps[i][tuple([coord_x[i], coord_y[i]])] for i in range(n_feat_maps)])
    pts_feat = pts_feat.reshape(n_batch, maps_per_batch, -1, k_closest, n_features)
    pts_feat = pts_feat.permute(0, 2, 3, 1, 4)
    # Sum all pts_feat from all feature maps
    # pts_feat = pts_feat.sum(dim=1)
    # pts_feat = torch.max(pts_feat, dim=1)[0]

    return pts_feat


def lin_weighting(z, distance, projected, my=0.9):


    inv_pt_ray_dist = torch.div(1, distance)
    pt_ray_dist_weights = inv_pt_ray_dist / torch.norm(inv_pt_ray_dist, dim=-1)[..., None]

    inv_proj_dist = torch.div(1, projected)
    pt_proj_dist_weights = inv_proj_dist / torch.norm(inv_proj_dist, dim=-1)[..., None]

    z = z * (my * pt_ray_dist_weights + (1-my) * pt_proj_dist_weights)[..., None, None]

    return torch.sum(z, dim=2)