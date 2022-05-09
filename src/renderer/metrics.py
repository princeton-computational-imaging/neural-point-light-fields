import numpy as np
import torch
import torch.nn as nn
from ..scenes import NeuralScene
from pytorch3d.renderer.implicit.utils import RayBundle


def calc_mse(x: torch.Tensor, y: torch.Tensor, **kwargs):
    return torch.mean((x - y) ** 2)


def calc_psnr(x: torch.Tensor, y: torch.Tensor, **kwargs):
    mse = calc_mse(x, y)
    psnr = -10.0 * torch.log10(mse)
    return psnr


def calc_latent_dist(xycfn, scene, reg, **kwargs):
    trainable_latent_nodes_id = scene.getSceneIdByTypeId(list(scene.nodes['scene_object'].keys()))

    # Remove "non"-nodes from the rays
    latent_nodess_id = xycfn[..., 4].unique().tolist()
    try:
        latent_nodess_id.remove(-1)
    except:
        pass

    # Just include nodes that have latent arrays
    # TODO: Do that for each class separatly
    latent_nodess_id = set(trainable_latent_nodes_id) & set(latent_nodess_id)

    if len(latent_nodess_id) != 0:
        latent_codes = torch.stack([scene.getNodeBySceneId(i).latent for i in latent_nodess_id])
        latent_dist = torch.sum(reg * torch.norm(latent_codes, dim=-1))
    else:
        latent_dist = torch.tensor(0.)

    return latent_dist


def get_rgb_gt(rgb: torch.Tensor, scene: NeuralScene, xycfn: torch.Tensor):
    xycf = xycfn[..., -1, :4].reshape(len(rgb), -1)
    rgb_gt = torch.zeros_like(rgb)

    # TODO: Make more efficient by not retriving image for each pixel,
    #  but storing all gt_images in a single tensor on the cpu
    # TODO: During test time just get all images avilable
    for f in xycf[:, 3].unique():
        if f == -1:
            continue
        frame = scene.frames[int(f)]
        for c in xycf[:, 2].unique():
            cf_mask = torch.all(xycf[:, 2:] == torch.tensor([c, f], device=xycf.device), dim=1)
            xy = xycf[cf_mask, :2].cpu()

            c_id = scene.getNodeBySceneId(int(c)).type_idx
            gt_img = frame.images[c_id]
            gt_px = torch.from_numpy(gt_img[xy[:, 1], xy[:, 0]]).to(device=rgb.device, dtype=rgb.dtype)
            rgb_gt[cf_mask] = gt_px

    return rgb_gt