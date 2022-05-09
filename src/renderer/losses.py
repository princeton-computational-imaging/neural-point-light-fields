import time

import numpy as np
import torch
import torch.nn as nn
from ..scenes import NeuralScene
from pytorch3d.renderer.implicit.utils import RayBundle
import imageio


def mse_loss(x, y, colorspace="RGB"):
    if "HS" in colorspace:
        # x,y = rgb2hsv(x),rgb2hsv(y)
        x, y = rgb2NonModuloHSV(x), rgb2NonModuloHSV(y)
        if colorspace == "HS":
            x, y = x[:, :-1], y[:, :-1]
    return torch.nn.functional.mse_loss(x, y)


def calc_uncertainty_weighted_loss(
    x: torch.Tensor, y: torch.Tensor, uncertainty: torch.Tensor, **kwargs
):
    uncertainty = uncertainty + 0.1
    uncertainty_variance = uncertainty ** 2
    # TODO: Check if mean or sum in loss function
    uncert_weighted_mse = torch.mean((x - y) ** 2 / (2 * uncertainty_variance))
    uncert_loss = (
        torch.mean(torch.log(uncertainty_variance) / 2) + 3
    )  # +3 to make it positive
    return uncert_weighted_mse + uncert_loss


def calc_density_regularizer(densities: torch.Tensor, reg: torch.int, **kwargs):
    return reg * torch.mean(densities)


def calc_weight_regularizer(
    weights: torch.Tensor,
    reg: torch.int,
    xycfn: torch.Tensor,
    scene: NeuralScene,
    **kwargs,
):
    background_weights = []

    n_rays, n_sampling_pts = weights.shape
    short_xycfn = xycfn[..., -n_sampling_pts:, :]
    short_xycfn = short_xycfn.reshape(n_rays, n_sampling_pts, -1)

    camera_ids_sampled = short_xycfn[:, 0, 2].unique()
    for cf in short_xycfn[:, 0, 2:4].unique(dim=0):
        c = int(cf[0])
        f = int(cf[1])
        frame = scene.frames[int(f)]

        cf_mask = torch.all(short_xycfn[..., 2:4] == cf, dim=-1)
        cf_mask = torch.where(cf_mask[:, 0])[0]

        relevant_xy = short_xycfn[cf_mask, 0, :2]
        xy_mask = tuple(relevant_xy[:, [1, 0]].T)
        segmentation_mask = frame.load_mask(c, return_xycfn=False)[xy_mask] == 0
        if xy_mask[0].shape[0] == 1:
            continue
        background_weights.append(weights[cf_mask[segmentation_mask]])

    background_weights = torch.cat(background_weights)

    return reg * torch.mean(background_weights)


def xycfn2list(xycfn):
    xycfn_list = []
    xycfn = xycfn.squeeze(0)
    for cf in xycfn[:, 0, 2:4].unique(dim=0):
        c = int(cf[0])
        f = int(cf[1])

        cf_mask = torch.all(xycfn[..., 2:4] == cf, dim=-1)
        cf_mask = torch.where(cf_mask[:, 0])[0]

        yx = xycfn[cf_mask, 0, :2]
        xycfn_list += [{"f": f, "c": c, "yx": yx}]

    return xycfn_list


def calc_psnr(x: torch.Tensor, y: torch.Tensor):
    mse = torch.nn.functional.mse_loss(x, y)
    psnr = -10.0 * torch.log10(mse)
    return psnr


# def calc_latent_dist(
#     xycfn, scene, reg, transient_frame_embbeding_reg=0, scene_function=None
# ):
#     loss = 0

#     trainable_latent_nodes_id = list(scene.nodes["scene_object"].keys())

#     # Remove "non"-nodes from the rays
#     latent_nodess_id = xycfn[..., 4].unique().tolist()
#     try:
#         latent_nodess_id.remove(-1)
#     except:
#         pass

#     # Just include nodes that have latent arrays
#     latent_nodess_id = set(trainable_latent_nodes_id) & set(latent_nodess_id)

#     if len(latent_nodess_id) != 0:
#         latent_codes = torch.stack(
#             [
#                 torch.cat(list(scene.nodes[i]["node"].latent.values()))
#                 for i in latent_nodess_id
#             ]
#         )
#         latent_dist = torch.sum(reg * torch.norm(latent_codes, dim=-1))
#     else:
#         latent_dist = torch.tensor(0.0)

#     loss += latent_dist

#     if transient_frame_embbeding_reg:
#         image_latent = []
#         fn = xycfn[..., 3:5].view(-1, 2).unique(dim=0)

#         tranient_latents = []
#         for (f, n) in fn:
#             # Get Transient embeddings for each node and frame combination
#             key = f"{f}_{n}"
#             if key in scene_function.transient_object_embeddings:
#                 fn_idx = torch.where(
#                     torch.all(fn == torch.tensor([f, n], device=fn.device), dim=1)
#                 )
#                 transient_embedding_object_frame = (
#                     scene_function.transient_object_embeddings[key]
#                 )
#                 tranient_latents.append(transient_embedding_object_frame)

#         # Find a loss on all those to be similar
#         image_latent = torch.stack(tranient_latents)
#         loss += transient_frame_embbeding_reg * torch.sum(
#             torch.std(image_latent, dim=0)
#         )

#     return loss


def extract_objects_HS_stats(unique_obj_IDs, RGB, per_pix_obj_ID, Hue2Cartesian=True):
    # It is actually not neccesary to do this in PyTorch.  Currently we are using GT colors, so we might as well work in Numpy since we don't need to backpropagate through it.
    if Hue2Cartesian:
        color_vals = rgb2NonModuloHSV(RGB)[:, :-1]
    else:
        color_vals = rgb2hsv(RGB)[:, :-1]
    per_obj_vals = dict(
        zip(
            unique_obj_IDs, [color_vals[per_pix_obj_ID == i, :] for i in unique_obj_IDs]
        )
    )
    # per_obj_color_stats = dict([(k,(np.mean(v.cpu().numpy(),0),np.cov(v.cpu().numpy().transpose()))) for k,v in per_obj_vals.items()])
    # color_means,color_covs = np.stack([per_obj_color_stats[k][0] for k in unique_obj_IDs]),np.stack([per_obj_color_stats[k][1] for k in unique_obj_IDs])
    per_obj_color_stats = dict(
        [(k, (torch.mean(v, 0), cov(v))) for k, v in per_obj_vals.items()]
    )
    color_means, color_covs = torch.stack(
        [per_obj_color_stats[k][0] for k in unique_obj_IDs]
    ), torch.stack([per_obj_color_stats[k][1] for k in unique_obj_IDs])
    return color_means, color_covs


def calc_latent_color_loss(unique_obj_IDs, RGB, per_pix_obj_ID, extract_latent_fn):
    color_means, color_covs = extract_objects_HS_stats(
        unique_obj_IDs, RGB, per_pix_obj_ID
    )
    MEANS_WEIGHT = 0.5
    latent_dists, color_dists = [], []
    for ind1 in range(len(unique_obj_IDs)):
        for ind2 in range(ind1 + 1, len(unique_obj_IDs)):
            latent_dist = torch.norm(
                extract_latent_fn(ind1) - extract_latent_fn(ind2), p=2, dim=0
            )  # /np.sqrt(latent_size)
            latent_dists.append(latent_dist)
            # color_dist = MEANS_WEIGHT*np.linalg.norm(color_means[ind1,:]-color_means[ind2,:],ord=2)/np.sqrt(color_means.shape[1])
            # color_dist += (1-MEANS_WEIGHT)*np.linalg.norm(color_covs[ind1,...]-color_covs[ind2,...],ord='fro')/color_means.shape[1]
            color_dist = MEANS_WEIGHT * torch.norm(
                color_means[ind1, :] - color_means[ind2, :], p=2, dim=0
            )  # /np.sqrt(color_means.shape[1])
            color_dist += (1 - MEANS_WEIGHT) * torch.norm(
                color_covs[ind1, ...] - color_covs[ind2, ...], p="fro"
            )  # /color_means.shape[1]
            color_dists.append(color_dist)
    latent_dists = torch.stack(latent_dists)
    color_dists = torch.stack(color_dists)
    # color_dists = np.array(color_dists)/np.mean(color_dists)*torch.mean(latent_dists).item()
    # return torch.mean((torch.tensor(color_dists).type(latent_dists.type())-latent_dists)**2)
    color_dists = color_dists / torch.mean(color_dists) * torch.mean(latent_dists)
    return torch.mean((color_dists - latent_dists) ** 2)


def prod_density_distribution(
    obj_weights: torch.Tensor, transient_weights: torch.Tensor
):
    obj_weights = obj_weights + 1e-5
    obj_sample_pdf = obj_weights / torch.sum(obj_weights, -1, keepdim=True)

    transient_weights = transient_weights + 1e-5
    transient_sample_pdf = transient_weights / torch.sum(
        transient_weights, -1, keepdim=True
    )

    return torch.mean(100 * obj_sample_pdf * transient_sample_pdf)


def get_rgb_gt(
    rgb: torch.Tensor, scene: NeuralScene, xycfn: torch.Tensor, use_gt_mask=False
):
    xycf = xycfn[..., 0, :4].reshape(len(rgb), -1)
    rgb_gt = torch.ones_like(rgb) * -1

    # TODO: Make more efficient by not retriving image for each pixel,
    #  but storing all gt_images in a single tensor on the cpu
    # TODO: During test time just get all images avilable
    camera_ids_sampled = xycf[:, 2].unique()
    for f in xycf[:, 3].unique():
        frame = scene.frames[int(f)]
        for c in frame.camera_ids:
            if c not in camera_ids_sampled:
                continue
            cf_mask = torch.all(
                xycf[:, 2:] == torch.tensor([c, f], device=xycf.device), dim=1
            )
            xy = xycf[cf_mask, :2].cpu()

            gt_img = frame.load_image(int(c))
            if use_gt_mask == 2:
                gt_img[frame.load_mask(c, return_xycfn=False) == 0] = 0
            if use_gt_mask == 3:
                gt_img[frame.load_mask(c, return_xycfn=False) == 0] = 1
            gt_px = torch.from_numpy(gt_img[xy[:, 1], xy[:, 0]]).to(
                device=rgb.device, dtype=rgb.dtype
            )
            rgb_gt[cf_mask] = gt_px

    assert (rgb_gt == -1).sum() == 0
    return rgb_gt


def rgb2hsv(rgb_vect):
    img = rgb_vect  # * 0.5 + 0.5
    per_pix_max, per_pix_min = img.max(1)[0], img.min(1)[0]
    delta = per_pix_max - per_pix_min
    delta_is_0 = delta == 0
    hue = torch.zeros([img.shape[0]]).to(img.device)
    max_is_R = torch.logical_and(
        img[:, 0] == per_pix_max, torch.logical_not(delta_is_0)
    )
    max_is_G = torch.logical_and(
        img[:, 1] == per_pix_max, torch.logical_not(delta_is_0)
    )
    max_is_B = torch.logical_and(
        img[:, 2] == per_pix_max, torch.logical_not(delta_is_0)
    )
    hue[max_is_B] = 4.0 + ((img[max_is_B, 0] - img[max_is_B, 1]) / delta[max_is_B])
    hue[max_is_G] = 2.0 + ((img[max_is_G, 2] - img[max_is_G, 0]) / delta[max_is_G])
    hue[max_is_R] = (
        0.0 + ((img[max_is_R, 1] - img[max_is_R, 2]) / delta[max_is_R])
    ) % 6

    hue[delta_is_0] = 0.0
    hue = hue / 6

    saturation = torch.zeros_like(hue)
    max_is_0 = per_pix_max == 0
    saturation[torch.logical_not(max_is_0)] = (
        delta[torch.logical_not(max_is_0)] / per_pix_max[torch.logical_not(max_is_0)]
    )
    saturation[max_is_0] = 0.0

    value = per_pix_max
    return torch.stack([hue, saturation, value], 1)


def hue2cartesian(hue):
    return 0.5 * torch.stack(
        [torch.cos(2 * np.pi * hue), torch.sin(2 * np.pi * hue)], 1
    )


def rgb2NonModuloHSV(rgb_vect):
    HSV = rgb2hsv(rgb_vect)
    return torch.cat([hue2cartesian(HSV[:, 0]), HSV[:, 1:]], 1)


def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    # From https://github.com/pytorch/pytorch/issues/19037
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w / w_sum)[:, None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()
