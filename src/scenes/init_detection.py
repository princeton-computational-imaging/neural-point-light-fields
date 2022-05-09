import numpy as np
import torch
from . import createSceneObject
import random


def init_frames_on_BEV_anchors(scene,
                               image_ls,
                               exp_dict_pretrained,
                               tgt_frames,
                               n_anchors_depth=3,
                               n_anchors_angle=3,
                               ):
    # Get camera
    cam = list(scene.nodes['camera'].values())[0]
    far = exp_dict_pretrained['scenes'][0]['far_plane']
    near = exp_dict_pretrained['scenes'][0]['near_plane']
    box_scale= exp_dict_pretrained['scenes'][0]['box_scale']
    cam_above_ground_kitti = 1.5

    param_dict = {}

    # Initialize frame with camera and background objects
    frame_idx = scene.init_blank_frames(image_ls,
                                        camera_node=cam,
                                        far=far,
                                        near=near,
                                        box_scale=box_scale,)

    # Create anchors at the midpoints of each cell in an BEV grid inside the viewing frustum
    anchor_angle, anchor_depth, anchor_height = _create_cyl_mid_point_anchors(cam,
                                                                              cam_above_ground_kitti,
                                                                              n_anchors_angle,
                                                                              n_anchors_depth,
                                                                              near,
                                                                              far)

    # Get known object paramteres to sample from
    known_objs = list(scene.nodes['scene_object'].keys())

    # Get Car object
    for v in scene.nodes['object_class'].values():
        if v.name == 'Car':
            obj_class = v

    for fr_id in frame_idx:
        # Add random deviations
        anchor_angle_fr, anchor_depth_fr = _add_random_offset_anchors(cam, anchor_angle, anchor_depth, n_anchors_angle,
                                                                      n_anchors_depth, near, far)

        # Combine angle and depth values to xyz
        anchor_x = anchor_depth * torch.tan(anchor_angle_fr)
        anchors = torch.cat([anchor_x[..., None], anchor_height, anchor_depth_fr[..., None]], dim=2)

        # Loop over all anchors
        for i, anchor in enumerate(anchors.view(-1, 3)):
            # Get params and add a new obj at this anchor
            size, latent = _get_anchor_obj_params(scene, known_objs)

            new_obj_dict = createSceneObject(length=size[0],
                                    height=size[1],
                                    width=size[2],
                                    object_class_node=obj_class,
                                    latent=latent,
                                    type_idx=i)
                                    
            nodes = scene.updateNodes(new_obj_dict)
            new_obj = list(nodes['scene_object'].values())[0]

            # Get rotation
            rotation = _get_anchor_box_rotation(anchor)

            # Add new edge for the anchor and new object to the scene
            scene.add_new_obj_edges(frame_id=fr_id,
                                    object_node=new_obj,
                                    translation=anchor,
                                    rotation=rotation,
                                    box_size_scaling=box_scale,)

        frame = scene.frames[fr_id]
        for obj_id in frame.get_objects_ids():
            for v_name, v in frame.get_object_parameters(obj_id).items():
                if v_name != 'scaling':
                    param_dict['{}_{}'.format(v_name, obj_id)] = v

        tgt_dict = {}
        for tgt_id in tgt_frames:
            tgt_frame = scene.frames[tgt_id]
            for obj_id in tgt_frame.get_objects_ids():
                for v_name, v in tgt_frame.get_object_parameters(obj_id).items():
                    if v_name != 'scaling':
                        tgt_dict['{}_{}'.format(v_name, obj_id)] = v


    camera_idx = [cam.scene_idx] * len(frame_idx)
    return param_dict, frame_idx, camera_idx, tgt_dict


def _create_cyl_mid_point_anchors(camera,
                                  cam_above_ground,
                                  n_anchors_angle,
                                  n_anchors_depth,
                                  near,
                                  far):
    # Basic BEV Anchors
    cam_param = camera.intrinsics
    fov_y = 2 * torch.arctan(cam_param.H / (2 * cam_param.f_y))
    fov_x = 2 * torch.arctan(cam_param.W / (2 * cam_param.f_x))

    device = fov_y.device

    # Sample anchors on BEV Plane inside camera viewing frustum
    # Sample anchors from left to right along the angle inside the FOV
    percent_fov_x = torch.linspace(0, n_anchors_angle - 1, n_anchors_angle, device=device) / n_anchors_angle
    angle_mid_point = 1 / (2 * n_anchors_angle)
    anchor_angle = fov_x * (percent_fov_x + angle_mid_point) - fov_x / 2
    anchor_angle = anchor_angle[None, :].repeat(n_anchors_depth, 1)

    # Sample along depth from near to far
    depth_mid_point = (far - near) / (2 * n_anchors_depth)
    anchor_depth = (far - near) * (
            torch.linspace(0, n_anchors_depth - 1, n_anchors_depth, device=device) / n_anchors_depth)
    anchor_depth += near + depth_mid_point
    anchor_depth = anchor_depth[:, None].repeat(1, n_anchors_angle)

    anchor_height = torch.ones(size=(n_anchors_depth, n_anchors_angle, 1), device=device) * cam_above_ground

    return anchor_angle, anchor_depth, anchor_height


def _add_random_offset_anchors(camera, anchor_angle, anchor_depth, n_anchors_angle, n_anchors_depth, near, far):
    # Basic BEV Anchors
    cam_param = camera.intrinsics
    fov_y = 2 * torch.arctan(cam_param.H / (2 * cam_param.f_y))
    fov_x = 2 * torch.arctan(cam_param.W / (2 * cam_param.f_x))

    device = anchor_angle.device

    # Add random deviations
    rand_angle = (2 * torch.rand(size=anchor_angle.shape, device=device) - 1) * (
            fov_x / n_anchors_angle)
    rand_depth = (2 * torch.rand(size=anchor_angle.shape, device=device) - 1) * (
            (far - near) / n_anchors_depth)

    anchor_angle_fr = anchor_angle + rand_angle
    anchor_depth_fr = anchor_depth + rand_depth
    return anchor_angle_fr, anchor_depth_fr


def _get_anchor_obj_params(scene, known_objs):
    random.shuffle(known_objs)
    known_obj_id = known_objs[0]
    known_obj = scene.nodes['scene_object'][known_obj_id]
    size = known_obj.box_size
    latent = known_obj.latent

    return size, latent


def _get_anchor_box_rotation(anchor):
    yaw = torch.rand((1,), device=anchor.device) * - np.pi / 2
    if yaw < 1e-4:
        yaw += 1e-4
    rotation = torch.tensor([0., yaw, 0.])
    return rotation