# from .nerf_tf.prepare_input_helper import plane_bounds
import copy
import time, torch
import numpy as np
import cProfile, pstats
from . import utils as ut
from . import utils_dist as utd
import tqdm
from PIL import Image
import os

import pandas as pd
from haven import haven_utils as hu
import imageio
import random
from src import renderer
import matplotlib.pyplot as plt
from haven import haven_img as hi
from src.scenes import get_obj_inputs
from pytorch3d.renderer.implicit.utils import RayBundle
from sklearn.decomposition import PCA

class Model:
    def __init__(
        self,
        scene,
        exp_dict,
        precache=True,
        device="cuda",
        args=None,
    ):
        self.device = device

        ### Point Cloud Debugging only, will not work with gt_mask
        self.overfit = exp_dict.get("overfit", None)
        self.render_cams = None

        self.lightfield_config = exp_dict.get("lightfield", None)
        self.exp_dict = exp_dict
        self.scene = scene

        self.renderer = renderer.PointLightFieldRenderer(
            scene=scene,
            chunk_size_test=exp_dict['chunk'],
            lightfield_config=exp_dict['lightfield'],
            point_chunk_size=exp_dict.get('point_chunk', 1e12),
            device=self.device,
        )

        # self.renderer.to(self.device)
        # if args is not None and args.distributed:
        #     self.renderer_with_ddp = torch.nn.parallel.DistributedDataParallel(
        #         self.renderer, device_ids=[args.gpu], find_unused_parameters=True
        #     )
        #     self.renderer = self.renderer_with_ddp.module
        # else:
        #     self.renderer_with_ddp = self.renderer

        self.renderer.neural_renderer.to(self.device)
        if False and (args is not None and args.distributed):
            self.renderer.neural_renderer_with_ddp = (
                torch.nn.parallel.DistributedDataParallel(
                    self.renderer.neural_renderer,
                    device_ids=[args.gpu],
                    find_unused_parameters=True,
                )
            )
            self.renderer.neural_renderer = self.renderer_with_ddp.module
        else:
            self.renderer.renderer_with_ddp = self.renderer.neural_renderer

        # Init Optimizer and scheduler
        if not scene.refine_camera_pose:
            self.opt = torch.optim.Adam(
                [p for p in self.renderer.parameters() if p.requires_grad == True],
                lr=exp_dict["lrate"],
            )
        else:
            self.opt = torch.optim.Adam(
                [p[1] for p in self.renderer.named_parameters() if not ('delta_translation' in p[0] or 'delta_rotation' in p[0]) and p[1].requires_grad == True],
                lr=exp_dict["lrate"],
            )
            self.calib_opt = torch.optim.Adam(
                [p for p in self.renderer.parameters() if p.requires_grad == True],
                lr=exp_dict["lrate"],
            )


        # TODO: Add precache rays to the exp_dict or args
        # precache_rays = exp_dict.get('precache', True)
        precache_intersections = False

        if precache:
            self.renderer.eval()
            # TODO: Add individual precaching as wrapper function to _scene_function again
            self.renderer.scene_renderer.raysampler.precache_rays(
                scene=scene,
                frame_indexes=scene.frame_ids,
                intersection_caching=precache_intersections,
            )

        # make sure number of latents is equal to the number of objects in the scene
        # assert len(self.renderer._latent_codes) == len(scene.nodes["scene_object"])
        self.epoch = 0

        self.renderer.to(self.device)

        # Keep poses on CPU
        self.renderer._poses.to('cpu')


    def set_state_dict(self, state_dict):
        self.renderer.load_state_dict(state_dict["renderer"], strict=False)
        self.opt.load_state_dict(state_dict["opt"])
        self.epoch = state_dict["epoch"]
        # self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])

    def get_state_dict(self):
        return {
            "renderer": self.renderer.state_dict(),
            "opt": self.opt.state_dict(),
            "epoch": self.epoch
            # 'lr_scheduler':self.lr_scheduler.state_dict()
        }

    def train_on_scene(self, scene_loader, calib_opt=False, bi=0):
        self.renderer.train()
        # create a random sampler of "k" batches
        # TODO: Instead of random sampling frames, make sure to iterate over all frames and rays in one epoch!

        # scene = scene_loader.dataset
        pbar = tqdm.tqdm(scene_loader)

        # go through the scene batches
        score_list = []
        for b in pbar:
            # TODO: Get the overfitting functionality back
            if self.overfit == "image":
                b = ut.collate_fn_dict_of_lists(
                    [scene_loader.dataset.__getitem__(bi, intersections_only=False, random_rays=False)]
                )
                b['frame_id'] = [0]
                b['camera_id'] = [1]
            elif self.overfit == "frame":
                b['frame_id'] = [0, 0]
                b['camera_id'] = [1, 2]
            
            gt = torch.cat(b["images"])[..., :3].cuda()
            s_time = time.time()
            rgb_out = self.renderer.forward_on_batch(b)
            e_time = time.time() - s_time
            pred = rgb_out["rgb"][..., :3]

            # compute loss
            loss_dict = self.renderer.compute_loss(
                rgb_out=rgb_out,
                gt=gt,
                xycfn=rgb_out.get('xycfn', None),
                weighted=self.exp_dict.get("weighted"),
            )
            psnr = self.renderer.compute_psnr(pred=pred, gt=gt, rgb_out=rgb_out)

            # take a training step.
            if not calib_opt:
                self.opt.zero_grad()
                loss = sum([l for l in loss_dict.values()])
                loss.backward()
                self.opt.step()
            else:
                self.calib_opt.zero_grad()
                loss = sum([l for l in loss_dict.values()])
                loss.backward()
                self.calib_opt.step()

            # record score
            pbar.set_description(f"loss: {float(loss):.2f} - psnr: {float(psnr):.2f} - forward: {e_time:.2f} (s)")
            score_dict = {"train_loss": loss, "train_psnr": psnr}
            for k, v in loss_dict.items():
                score_dict[k] = float(loss_dict[k])
            score_list += [score_dict]

        self.epoch += 1
        train_dict = pd.DataFrame(score_list).mean().to_dict()
        return train_dict

    @torch.no_grad()
    def val_on_scene(self, scene, savedir_images, random=True, new_path=False, offset_traj=False, all_frames=False, EPI=False,
                     epi_row=None, epi_frame_idx=None):
        self.renderer.eval()
        # get sequential loader on a subset
        ind_list = list(np.arange(len(scene)))
        if new_path:
            # TODO: Allow different cameras
            scene, ind_list = ut.add_camera_path_frame(scene,
                                                       frame_idx=list(np.random.choice(ind_list, 2)),
                                                       cam_idx=[1,1],
                                                       n_steps=12,)
        elif offset_traj:
            scene, ind_list = ut.add_camera_path_frame(scene,
                                                       frame_idx=[ind_list[0],
                                                                  ind_list[len(ind_list) // 4],
                                                                  ind_list[len(ind_list) // 2],
                                                                  ind_list[3 * len(ind_list) // 4],
                                                                  ind_list[-1]],
                                                       cam_idx=[1, 1, 1, 1, 1],
                                                       n_steps=len(ind_list)//6,
                                                       offset=1.,
                                                       )
            import matplotlib.pyplot as plt
            trans_data = torch.cat([fr.edges[0].translation for fr in scene.frames.values()])[:-1].numpy()
            trans_new = torch.cat([ed.translation for ed in list(scene.frames[41].edges.values())[22:]]).numpy()

            plt.scatter(trans_data[:, 0], trans_data[:, 1])
            plt.scatter(trans_new[:, 0], trans_new[:, 1])

        elif all_frames:
            ind_list = list(np.arange(len(scene)))
        elif EPI and epi_frame_idx is not None:
            ind_list = [epi_frame_idx]
        elif random:
            ind_list = ind_list[0] + list(np.random.choice(ind_list[1:], 4))
        else:
            ind_list = ind_list[0]

        # subset = torch.utils.data.Subset(scene, ind_list)
        pbar = tqdm.tqdm(ind_list)
        # pbar = tqdm.tqdm([0] + list(ind_list))
        # pbar = tqdm.tqdm([0])
        # b_list = ut.collate_fn_dict_of_lists(subset)

        # go through the scene batches
        for bck_only, obj_only in [
            # (True, True),
            (False, True),
            # (False, False)
        ]:
            val_list = []
            for i, bi in enumerate(pbar):
                b = ut.collate_fn_dict_of_lists(
                    [scene.__getitem__(bi, intersections_only=False, random_rays=False, validation=True, EPI=EPI,
                                       epi_row=epi_row)]
                )

                val_dict = self.val_on_batch(
                    b,
                    savedir_images,
                    bck_only=bck_only,
                    obj_only=obj_only,
                    image_id=i,
                    file_identifier=0,
                )
                psnr = val_dict["psnr"]
                pbar.set_description(f"psnr: {float(psnr):.2f}")
                # record score

                val_list += [{f"val_{k}": v for k, v in val_dict.items()}]

                if self.overfit == "frame":
                    ut.output_gif(scene, ind_list, savedir_images, tgt_fname='overfit_frame')
                          
        val_dict = pd.DataFrame(val_list).mean().to_dict()

        if new_path:
            # Clean up after creating random path
            novel_frame_id = scene.frames_cameras[ind_list[0]][0]
            novel_cameras = scene.frames[novel_frame_id].camera_ids
            for c_id in novel_cameras:
                del scene.nodes['camera'][c_id]
            del scene.frames[novel_frame_id]

            scene.frames_cameras = scene.frames_cameras[:-len(ind_list)]

        return val_dict

    def forward_on_batch(self, batch):
        input_dict = self.renderer.forward_on_batch(batch, device=self.device)

        rgb_out = self.renderer_with_ddp(
            scene=self.scene,
            input_dict=input_dict,
            threshold_alpha=None,
        )

        rgb_out["object_latent"] = input_dict["object_latent"]
        return rgb_out

    @torch.no_grad()
    def predict_on_batch(self, batch):
        self.renderer.eval()

        # for bs in range(len())
        rgb_out = self.renderer.forward_on_batch(batch)

        H, W = batch["H"][0], batch["W"][0]
        rgb_out["rgb"] = rgb_out["rgb"].squeeze().reshape(H, W, 3)
        return rgb_out

    @torch.no_grad()
    def val_on_batch(
        self, b, savedir_images, bck_only=False, obj_only=True, image_id=0,file_identifier=None
    ):
        # compute predictions
        ray_bundle = b["ray_bundle"][0]
        
        rgb_out = self.renderer.forward_on_batch(b)
        # stats = pstats.Stats(pr).sort_stats("cumtime")

        # Print the stats report
        # stats.print_stats()

        val_dict = {}
        pred = rgb_out["rgb"][..., :3]
        gt = b["images"][0][..., :3]

        # save predictions
        H, W = b["H"][0], b["W"][0]
        # #######
        # # EPI-Polar 100 Lines (+-50)
        # epi_row = 96
        # plt.figure()
        # epi = pred.squeeze(0).cpu().reshape(101, W, 3).numpy()
        # epi[50] += np.array([[0., 1., 9.]])
        # epi = np.minimum(epi, 1.)
        # plt.imshow(epi)
        #
        # gt_cp = copy.deepcopy(gt)
        # gt_cp[epi_row, ...] += torch.tensor([[0.,6.,0.]])
        # gt_cp = torch.min(gt_cp, torch.tensor(1.))
        # plt.figure()
        # plt.imshow(gt_cp)
        #
        # hu.save_image('/home/julian/Desktop/epis/GT.png', gt_cp)
        # hu.save_image('/home/julian/Desktop/epis/LF_slice.png', epi)
        # #######
        png_pred = pred.squeeze(0).cpu().reshape(H, W, 3)
        png_gt = gt.reshape(H, W, 3)

        psnr = self.renderer.compute_psnr(pred=png_pred, gt=png_gt, rgb_out=rgb_out)

        png_pred = png_pred.numpy()
        png_gt = png_gt.numpy()

        i1 = np.round(png_pred * 255).astype(np.uint8)

        png_gt = png_gt.clip(0, 1)

        i2 = np.round(png_gt * 255).astype(np.uint8)
        if utd.is_main_process():
            img_comp = np.hstack([i1, i2])
            icompname = f'scene_{b["meta"][0]["scene_id"]}_{b["frame_id"][0]}_{b["camera_id"][0]}'
            ipredname = f'frame_{str(b["frame_id"][0]).zfill(3)}_camera_{str(b["camera_id"][0]).zfill(2)}'
            # print(os.path.join(savedir_images, f"{iname}.png"))
            hu.save_image(os.path.join(savedir_images, f"{icompname}.png"), img_comp)
            hu.save_image(os.path.join(savedir_images, f"{ipredname}.png"), i1)
            val_dict = {}

        val_dict["psnr"] = psnr
        return val_dict


def to_device(ray_bundle, local_pts_bundle, device):
    local_pts_bundle = local_pts_bundle
    if local_pts_bundle[0] is not None:
        local_pts_bundle[0] = [l.to(device) for l in local_pts_bundle[0]]
        local_pts_bundle[1] = local_pts_bundle[1].to(device)
        local_pts_bundle[2] = local_pts_bundle[2].to(device)

    ray_bundle = ray_bundle._replace(
        lengths=ray_bundle.lengths.to(device),
        origins=ray_bundle.origins.to(device),
        directions=ray_bundle.directions.to(device),
        xys=ray_bundle.xys.to(device),
    )

    return (
        ray_bundle.lengths,
        ray_bundle.origins,
        ray_bundle.directions,
        ray_bundle.xys,
        local_pts_bundle,
    )