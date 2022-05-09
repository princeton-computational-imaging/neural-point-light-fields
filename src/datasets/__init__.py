import pylab as plt

# import tensorflow as tf
# from ..nerf_tf.prepare_input_helper import extract_object_information
import numpy as np
import torch, os, copy, glob

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataset(datadir, scene_dict, args):
    scene_dict = copy.deepcopy(scene_dict)
    selected_frames = None
    if scene_dict["last_frame"] is not None:
        selected_frames = [scene_dict["first_frame"], scene_dict["last_frame"]]

    if scene_dict['type'] == 'waymo':
        from .waymo import Waymo
        # Scene Id in Waymo is a combination of dataset number (e.g. 4 for 0004)
        # and the rank of the tfrecord sorted alphabetical (e.g. 0 for segment-18446264979321894359_3700_000_3720_000_with_camera_labels.tfrecord)
        datadir_full = datadir + ''
        scene_id = scene_dict['scene_id']
        record_id = scene_id[1]
        subdirs_list = [d.name for d in os.scandir(datadir) if d.is_dir() if d.name[-4:].isnumeric()]
        assert scene_id[0]<len(subdirs_list),'Folder %s does not have %d subfolers'%(datadir,scene_id[0]+1)
        # datadir_full = os.path.join(datadir, subdirs_list[scene_id[0]])
        for sub_dir in sorted(subdirs_list):
            if sub_dir[-4:].isnumeric():
                if int(sub_dir[-4:]) == scene_id[0]:
                    datadir_full = os.path.join(datadir, sub_dir)
                    datadir_full = sorted([f.path for f in os.scandir(datadir_full) if f.is_dir()])[record_id]
                    break
        # datadir_full = os.path.join(datadir_full, record_dir)

        dataset = Waymo(datadir_full, scene_dict, selected_frames=selected_frames,object_types=scene_dict['object_types'])

    else:
        raise ValueError("dataset does not exist")
    # print(f'Loaded {scene_dict["type"]}', dataset.images.shape,  dataset.hwf, datadir_full)
    print(
        "\nLoaded scene",
        os.path.split(datadir_full)[-1],
        f"with {len(dataset.images)} frames.",
    )
    # sanity plots
    # plot_poses(dataset, fname=f".tmp/poses_{scene_dict['type']}-{scene_dict['scene_id']}.png")
    # plot_poses(dataset.poses, dataset.visible_objects, dataset.object_positions, fname='.tmp/tmp.png')

    return dataset


def plot_bboxes(dataset, fname):
    poses, visible_objects, object_positions = (
        dataset.poses,
        dataset.visible_objects,
        dataset.object_positions,
    )


def plot_poses(dataset, fname):
    poses, visible_objects, object_positions = (
        dataset.poses,
        dataset.visible_objects,
        dataset.object_positions,
    )
    plot_poses = True
    plot_obj = True
    ax_birdseye = [0, -1]  # if args.dataset_type == 'vkitti' else [0,1]
    ax_zy = [-1, 1]  # if args.dataset_type == 'vkitti' else [1,2]
    ax_xy = [0, 1]  # if args.dataset_type == 'vkitti' else [0,2]
    if plot_poses:
        fig, ax_lst = plt.subplots(2, 2)

        plt.suptitle(os.path.split(fname)[-1])
        position = []
        for pose in poses:
            position.append(pose[:3, -1])

        position_array = np.array(position).astype(np.float32)
        plt.sca(ax_lst[0, 0])
        plt.scatter(
            position_array[:, ax_birdseye[0]],
            position_array[:, ax_birdseye[1]],
            color="b",
        )
        plt.sca(ax_lst[1, 0])
        plt.scatter(position_array[:, ax_xy[0]], position_array[:, ax_xy[1]], color="b")
        plt.sca(ax_lst[0, 1])
        plt.scatter(position_array[:, ax_zy[0]], position_array[:, ax_zy[1]], color="b")

        if plot_obj:
            if not object_positions.shape[1] == 0:
                object_positions = np.squeeze(
                    object_positions[np.argwhere(object_positions[:, 0] != -1)]
                )
                object_positions = (
                    object_positions[None, :]
                    if len(object_positions.shape) == 1
                    else object_positions
                )
                plt.sca(ax_lst[0, 0])
                plt.scatter(
                    object_positions[:, ax_birdseye[0]],
                    object_positions[:, ax_birdseye[1]],
                    color="black",
                )
                plt.sca(ax_lst[1, 0])
                plt.scatter(
                    object_positions[:, ax_xy[0]],
                    object_positions[:, ax_xy[1]],
                    color="black",
                )
                plt.sca(ax_lst[0, 1])
                plt.scatter(
                    object_positions[:, ax_zy[0]],
                    object_positions[:, ax_zy[1]],
                    color="black",
                )

                # Get locale coordinates of the very first object
                plt.sca(ax_lst[0, 0])
                headings = np.reshape(visible_objects[..., 10], [-1, 1])
                t_o_w = np.reshape(visible_objects[..., 7:10], [-1, 3])
                # theta_0 = visible_objects_plt[0, 0, 10]
                # r_ov_0 = np.array([[np.cos(theta_0), 0, np.sin(theta_0)], [0, 1, 0], [-np.sin(theta_0), 0, np.cos(theta_0)]])
                for i, yaw in enumerate(headings):
                    yaw = float(yaw)
                    r_ov = np.array(
                        [
                            [np.cos(yaw), 0, np.sin(yaw)],
                            [0, 1, 0],
                            [-np.sin(yaw), 0, np.cos(yaw)],
                        ]
                    )
                    t = t_o_w[i]
                    x_v = np.matmul(
                        np.concatenate([r_ov, t[:, None]], axis=1), [5.0, 0.0, 0.0, 1.0]
                    )
                    z_v = np.matmul(
                        np.concatenate([r_ov, t[:, None]], axis=1), [0.0, 0.0, 5.0, 1.0]
                    )
                    v_origin = np.matmul(
                        np.concatenate([r_ov, t[:, None]], axis=1), [0.0, 0.0, 0.0, 1.0]
                    )

                    plt.arrow(
                        v_origin[0],
                        v_origin[2],
                        x_v[0] - v_origin[0],
                        x_v[2] - v_origin[2],
                        color="black",
                        width=0.1,
                    )
                    plt.arrow(
                        v_origin[0],
                        v_origin[2],
                        z_v[0] - v_origin[0],
                        z_v[2] - v_origin[2],
                        color="orange",
                        width=0.1,
                    )

        # For waymo  x --> -z, y --> x, z --> y
        x_c_0 = np.matmul(poses[0, :, :], np.array([5.0, 0.0, 0.0, 1.0]))[:3]
        y_c_0 = np.matmul(poses[0, :, :], np.array([0.0, 5.0, 0.0, 1.0]))[:3]
        z_c_0 = np.matmul(poses[0, :, :], np.array([0.0, 0.0, 5.0, 1.0]))[:3]
        coord_cam_0 = [x_c_0, y_c_0, z_c_0]
        c_origin_0 = poses[0, :3, 3]

        plt.sca(ax_lst[0, 0])
        plt.arrow(
            c_origin_0[ax_birdseye[0]],
            c_origin_0[ax_birdseye[1]],
            coord_cam_0[ax_birdseye[0]][ax_birdseye[0]] - c_origin_0[ax_birdseye[0]],
            coord_cam_0[ax_birdseye[0]][ax_birdseye[1]] - c_origin_0[ax_birdseye[1]],
            color="red",
            width=0.1,
        )
        plt.arrow(
            c_origin_0[ax_birdseye[0]],
            c_origin_0[ax_birdseye[1]],
            coord_cam_0[ax_birdseye[1]][ax_birdseye[0]] - c_origin_0[ax_birdseye[0]],
            coord_cam_0[ax_birdseye[1]][ax_birdseye[1]] - c_origin_0[ax_birdseye[1]],
            color="green",
            width=0.1,
        )
        plt.axis("equal")
        plt.sca(ax_lst[1, 0])
        plt.arrow(
            c_origin_0[ax_xy[0]],
            c_origin_0[ax_xy[1]],
            coord_cam_0[ax_xy[0]][ax_xy[0]] - c_origin_0[ax_xy[0]],
            coord_cam_0[ax_xy[0]][ax_xy[1]] - c_origin_0[ax_xy[1]],
            color="red",
            width=0.1,
        )
        plt.arrow(
            c_origin_0[ax_xy[0]],
            c_origin_0[ax_xy[1]],
            coord_cam_0[ax_xy[1]][ax_xy[0]] - c_origin_0[ax_xy[0]],
            coord_cam_0[ax_xy[1]][ax_xy[1]] - c_origin_0[ax_xy[1]],
            color="green",
            width=0.1,
        )
        plt.axis("equal")
        plt.sca(ax_lst[0, 1])
        plt.arrow(
            c_origin_0[ax_zy[0]],
            c_origin_0[ax_zy[1]],
            coord_cam_0[ax_zy[0]][ax_zy[0]] - c_origin_0[ax_zy[0]],
            coord_cam_0[ax_zy[0]][ax_zy[1]] - c_origin_0[ax_zy[1]],
            color="red",
            width=0.1,
        )
        plt.arrow(
            c_origin_0[ax_zy[0]],
            c_origin_0[ax_zy[1]],
            coord_cam_0[ax_zy[1]][ax_zy[0]] - c_origin_0[ax_zy[0]],
            coord_cam_0[ax_zy[1]][ax_zy[1]] - c_origin_0[ax_zy[1]],
            color="green",
            width=0.1,
        )
        plt.axis("equal")

        # Plot global coord axis
        plt.sca(ax_lst[0, 0])
        plt.arrow(0, 0, 5, 0, color="cyan", width=0.1)
        plt.arrow(0, 0, 0, 5, color="cyan", width=0.1)
    plt.savefig(fname)
