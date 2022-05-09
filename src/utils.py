import tqdm, argparse
import os, exp_configs
import copy
import pandas as pd
import numpy as np
import torch
from PIL import Image
from haven import haven_utils as hu
from haven import haven_examples as he
from haven import haven_wizard as hw
from haven import haven_results as hr
from src import models
from src.scenes import NeuralScene
from .scenes import createCamera, createStereoCamera
# from src.renderer import NeuralSceneRenderer
from pytorch3d.transforms.rotation_conversions import axis_angle_to_quaternion, quaternion_to_axis_angle
import time
from pytorch3d.transforms.so3 import so3_exponential_map, so3_log_map
from pytorch3d.transforms.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
)


def mask_to_xycfn(frame_idx, camera_idx, mask):
    xycfn_list = []
    for node_id in np.unique(mask):
        if node_id == 0:
            continue
        y, x = np.where(mask == node_id)
        xycfn_i = np.zeros((y.shape[0], 5))
        xycfn_i[:, 4] = node_id
        xycfn_i[:, 3] = frame_idx
        xycfn_i[:, 2] = camera_idx
        xycfn_i[:, 1] = y
        xycfn_i[:, 0] = x
        xycfn_list += [xycfn_i]
    xycfn = np.vstack(xycfn_list)
    return xycfn


class to_args(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def collate_fn_lists_of_dicts(batch):
    return batch


def collate_fn_dict_of_lists(batch):
    import functools

    return {
        key: [item[key] for item in batch]
        for key in list(
            functools.reduce(
                lambda x, y: x.union(y), (set(dicts.keys()) for dicts in batch)
            )
        )
    }


def display_points(gt_image, out, color):
    obj_mask = torch.where(out["xycfn"][..., -1].ge(1))
    xycf_relevant = out["xycfn"][obj_mask][..., :4]
    for f in xycf_relevant[..., 3].unique():
        # frame = scene.frames[int(f)]
        for c in xycf_relevant[:, 2].unique():
            cf_mask = torch.all(
                xycf_relevant[:, 2:]
                == torch.tensor([c, f], device=xycf_relevant.device),
                dim=1,
            )
            xy = xycf_relevant[cf_mask, :2].cpu()
            # c_id = scene.getNodeBySceneId(int(c)).type_idx
            gt_img = gt_image.copy()
            gt_img[xy[:, 1], xy[:, 0]] = np.array(color)
    return gt_img


def add_camera_path_frame(scene, frame_idx=[0], cam_idx=None, n_steps=9, remove_old_cameras=True, offset=0.):
    edges_to_cameras_ls = []
    imgs = []
    frames = []
    cams = []

    if cam_idx is None:
        cam_idx = [None] * len(frame_idx)

    max_ind_list = len(scene.frames_cameras)

    for frame_id, c_id in zip(frame_idx, cam_idx):
        # Extract all information from the frame for which cameras should be interpolated
        selected_frame = scene.frames[frame_id]
        frames += [selected_frame]
        cam_nodes = scene.nodes['camera']
        if c_id is None:
            # KITIT
            cams += [list(cam_nodes.values())[0]]

            edges_to_cameras = selected_frame.get_edge_by_child_idx(list(cam_nodes.keys()))
            edges_to_cameras_ls += [ed[0] for ed in edges_to_cameras]

            # edge_ids_to_cameras_ls = selected_frame.get_edge_idx_by_child_idx(list(cam_nodes.keys()))
            # edge_ids_to_cameras_ls = [ed_id[0] for ed_id in edge_ids_to_cameras_ls]

            imgs += [list(selected_frame.images.values())[0]]
        else:
            cams += [cam_nodes[c_id]]

            edges_to_cameras = selected_frame.get_edge_by_child_idx([c_id])
            edges_to_cameras_ls += [ed[0] for ed in edges_to_cameras]

            # edge_ids_to_cameras_ls = selected_frame.get_edge_idx_by_child_idx([c_id])
            # edge_ids_to_cameras_ls = [ed_id[0] for ed_id in edge_ids_to_cameras_ls]

            imgs += [selected_frame.images[c_id]]

    scene_dict = {}
    for fr_c in scene.frames_cameras:
        if fr_c[0] == frame_id:
            scene_dict = fr_c[2]
            scene_id = scene_dict['scene_id']
            break

    add_new_val_render_path(frames, cams, edges_to_cameras_ls, imgs, scene, scene_dict, n_steps,
                             remove_old_cameras, offset)

    new_max_ind_list = len(scene.frames_cameras)
    ind_list = np.linspace(max_ind_list, new_max_ind_list - 1, new_max_ind_list - max_ind_list, dtype=int)

    return scene, ind_list


def add_new_val_render_path(frames, cams, edges_to_cameras_ls, img_path, scene, scene_dict, n_steps=3, remove_old_cameras=False,
                            offset=0.):
    selected_frame = frames[0]
    copied_cam = cams[0]
    cam_ids = scene.nodes['camera'].keys()

    fr_edge_ids_to_cameras_ls = [[fr.frame_idx, ed.index] for fr, ed in zip(frames, edges_to_cameras_ls)]
    # Create a new frame as a copy of the selected frame
    new_frame = copy.deepcopy(selected_frame)

    # Remove image paths from unused cameras
    if remove_old_cameras:
        for c_id in cam_ids:
            if c_id in new_frame.images:
                del new_frame.images[c_id]
            if len(new_frame.get_edge_by_child_idx([c_id])[0]) > 0:
                new_frame.camera_ids.remove(c_id)

    # Get Camera poses
    new_rotations, new_translations = interpolate_between_camera_edges(edges_to_cameras_ls, n_steps=n_steps, offset=offset)
    new_rotations_no, new_translations_no = interpolate_between_camera_edges(edges_to_cameras_ls, n_steps=n_steps,
                                                                       offset=0.)

    # Create edges and nodes from new cameras
    new_edges_ls = []
    for k, (rotation, translation) in enumerate(zip(new_rotations, new_translations)):
        # Create new virtual camera
        new_cam = createCamera(copied_cam.H, copied_cam.W, copied_cam.intrinsics.f_x.cpu().numpy(), type=scene_dict['type'])
        new_nodes = scene.updateNodes(new_cam)
        new_cam_id = list(new_nodes['camera'].keys())[0]
        new_frame.images[new_cam_id] = img_path[k // n_steps]
        new_frame.camera_ids.append(new_cam_id)

        # Create new edge to the camera
        new_edge = copy.deepcopy(edges_to_cameras_ls[0])
        new_edge.translation = translation
        new_edge.rotation = rotation
        new_edge.child = new_cam_id

        new_edges_ls.append(new_edge)

    # Remove old cameras if requested
    # if remove_old_cameras:
    #     for fr_id, ed_id in fr_edge_ids_to_cameras_ls:
    #         new_frame.removeEdge(ed_id)

    # Add edges to the new camera poses to the graph
    new_frame.add_edges(new_edges_ls)

    # Add frame to the scene
    frame_list = [new_frame]

    scene_id = 1e4
    if not len(scene_dict) == 0:
        scene_id = scene_dict['scene_id']

    scene.updateFrames(frame_list, scene_id, scene_dict)


def add_new_val_render_poses(frame_to_copy_from, camera_to_copy_from, edges_to_cameras_ls, img_path, scene, scene_dict, n_steps=3, remove_old_cameras=True):
    selected_frame = frame_to_copy_from
    copied_cam = camera_to_copy_from
    cam_ids = scene.nodes['camera'].keys()

    edge_ids_to_cameras_ls = selected_frame.get_edge_idx_by_child_idx(list(cam_ids))
    edge_ids_to_cameras_ls = [ed_id[0] for ed_id in edge_ids_to_cameras_ls]
    # Create a new frame as a copy of the selected frame
    new_frame = copy.deepcopy(selected_frame)

    # Remove image paths from unused cameras
    if remove_old_cameras:
        for c_id in cam_ids:
            del new_frame.images[c_id]
            new_frame.camera_ids.remove(c_id)

    # Get Camera poses
    new_rotations, new_translations = interpolate_between_camera_edges(edges_to_cameras_ls, n_steps=n_steps, )

    # Create edges and nodes from new cameras
    new_edges_ls = []
    for k, (rotation, translation) in enumerate(zip(new_rotations, new_translations)):
        # Create new virtual camera
        new_cam = createCamera(copied_cam.H, copied_cam.W, copied_cam.intrinsics.f_x.cpu().numpy(), )
        new_nodes = scene.updateNodes(new_cam)
        new_cam_id = list(new_nodes['camera'].keys())[0]
        new_frame.images[new_cam_id] = img_path
        new_frame.camera_ids.append(new_cam_id)

        # Create new edge to the camera
        new_edge = copy.deepcopy(edges_to_cameras_ls[0])
        new_edge.translation = translation
        new_edge.rotation = rotation
        new_edge.child = new_cam_id

        new_edges_ls.append(new_edge)

    # Remove old cameras if requested
    if remove_old_cameras:
        for ed_id in edge_ids_to_cameras_ls:
            new_frame.removeEdge(ed_id)

    # Add edges to the new camera poses to the graph
    new_frame.add_edges(new_edges_ls)

    # Add frame to the scene
    frame_list = [new_frame]

    scene_id = 1e4
    if not len(scene_dict) == 0:
        scene_id = scene_dict['scene_id']

    scene.updateFrames(frame_list, scene_id, scene_dict)


def interpolate_between_camera_edges(edges_to_cameras_ls, n_steps=5, offset=0.):
    rots = [ed.rotation for ed in edges_to_cameras_ls]
    translations = [ed.translation for ed in edges_to_cameras_ls]

    steps = torch.linspace(0, 1, n_steps, device=rots[0].device)

    new_quat_rots = []
    new_translations = []
    for cam_pair_i in range(len(rots) - 1):
        # Interpolation between translations
        translation_0 = translations[cam_pair_i] + torch.matmul(so3_exponential_map(rots[cam_pair_i]),
                                                                torch.tensor([1., 0., 0.]) * offset)
        translation_1 = translations[cam_pair_i + 1] + torch.matmul(so3_exponential_map(rots[cam_pair_i + 1]),
                                                                    torch.tensor([1., 0., 0.]) * offset)
        mid_translations = translation_0 * (1 - steps[:, None]) + \
                           translation_1 * (steps[:, None])

        for i in range(n_steps - 1):
            new_translations.append(mid_translations[i, None])

        # Implementation between rotations with Quaternion SLERP
        quat_rot_0 = axis_angle_to_quaternion(rots[cam_pair_i])
        quat_rot_1 = axis_angle_to_quaternion(rots[cam_pair_i + 1])
        cosHalfTheta = torch.sum(quat_rot_0 * quat_rot_1)
        halfTheta = torch.acos(cosHalfTheta)
        sinHalfTheta = torch.sqrt(1.0 - cosHalfTheta * cosHalfTheta)

        if (torch.abs(sinHalfTheta) < 0.001):
            # theta is 180 degree --> Rotation around different axis is possible
            for i in range(n_steps - 1):
                new_quat_rots.append(quat_rot_0)
        elif (torch.abs(cosHalfTheta) >= 1.0):
            # 0 degree difference --> No new rotation necessary
            mid_quat = quat_rot_0 * (1. - steps)[:, None] + quat_rot_1 * (steps)[:, None]
            for i in range(n_steps - 1):
                new_quat_rots.append(mid_quat[i, None])
        else:
            ratioA = torch.sin((1 - steps) * halfTheta) / sinHalfTheta
            ratioB = torch.sin(steps * halfTheta) / sinHalfTheta
            mid_quat = quat_rot_0 * ratioA[:, None] + quat_rot_1 * ratioB[:, None]
            for i in range(n_steps - 1):
                new_quat_rots.append(mid_quat[i, None])

    new_quat_rots.append(quat_rot_1)
    new_translations.append(translation_1)

    new_rotations = [quaternion_to_axis_angle(quaterion) for quaterion in new_quat_rots]

    return new_rotations, new_translations


def output_gif(scene, ind_list, savedir_images, tgt_fname):
    tgt_path = os.path.join(savedir_images, (tgt_fname + '.gif'))

    iname_ls = [f'frame_{scene.frames_cameras[i][0]}_camera_{scene.frames_cameras[i][1]}' for i in ind_list]

    image_path_ls = [os.path.join(savedir_images, f"{iname}.png") for iname in iname_ls]

    img, *imgs = [Image.open(f) for f in image_path_ls]
    img.save(
        fp=tgt_path,
        format="GIF",
        append_images=imgs,
        save_all=True,
        duration=3000 // len(image_path_ls),
        loop=0,
    )