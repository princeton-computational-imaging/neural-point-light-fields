import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image


def extract_waymo_poses(dataset, scene_list, i_train, i_test, i_all):
    cam_names = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
    assert len(scene_list) == 1
    first_fr = scene_list[0]['first_frame']
    last_fr = scene_list[0]['last_frame']

    pose_file_n = "poses_{}_{}.npy".format(str(first_fr).zfill(4), str(last_fr).zfill(4))
    img_file_n = "imgs_{}_{}.npy".format(str(first_fr).zfill(4), str(last_fr).zfill(4))
    index_test_n= "index_test_{}_{}.npy".format(str(first_fr).zfill(4), str(last_fr).zfill(4))
    index_train_n = "index_train_{}_{}.npy".format(str(first_fr).zfill(4), str(last_fr).zfill(4))
    pt_depth_file_n = "pt_depth_{}_{}.npy".format(str(first_fr).zfill(4), str(last_fr).zfill(4))
    depth_img_file_n = "pt_depth_{}_{}.npy".format(str(first_fr).zfill(4), str(last_fr).zfill(4))

    segmemt_pth = ''
    for s in dataset.images[0].split('/')[1:-2]:
        segmemt_pth += '/' + s

    remove_side_views = True

    n_frames = len(dataset.poses_world) // 5

    if remove_side_views:
        n_imgs = n_frames * 3
        cam_names = cam_names[:3]

    else:
        n_imgs = n_frames * 5

    cam_pose = dataset.poses_world[:n_imgs]
    img_path = dataset.images[:n_imgs]

    cam_pose_openGL = cam_pose.dot(np.array([[-1., 0., 0., 0., ], [0., 1., 0., 0., ], [0., 0., -1., 0., ], [0., 0., 0., 1., ], ]))

    # Add H, W, focal
    hwf1 = [np.array([[dataset.H[c_name], dataset.W[c_name], dataset.focal[c_name], 1.]]).repeat(n_frames, axis=0) for c_name in cam_names]
    hwf1 = np.concatenate(hwf1)[:, :, None]
    cam_pose_openGL = np.concatenate([cam_pose_openGL, hwf1], axis=2)

    np.save(os.path.join(segmemt_pth, pose_file_n), cam_pose_openGL)
    np.save(os.path.join(segmemt_pth, img_file_n), img_path)
    np.save(os.path.join(segmemt_pth, index_test_n), i_test)
    np.save(os.path.join(segmemt_pth, index_train_n), i_train)

    xyz_pts = []

    # Extract depth points (and images)
    for i in range(n_imgs):
        fr  = i % n_frames
        pts_i_veh = np.asarray(o3d.io.read_point_cloud(dataset.point_cloud_pth[fr]).points)
        pts_i_veh = np.concatenate([pts_i_veh, np.ones([len(pts_i_veh), 1])], axis=-1)

        cam2veh_i = dataset.poses[i]
        veh2cam_i = np.concatenate([cam2veh_i[:3, :3].T, np.matmul(cam2veh_i[:3, :3].T, -cam2veh_i[:3, 3])[:, None]], axis=-1)

        pts_i_cam = np.matmul(veh2cam_i, pts_i_veh.T).T

        focal_i = hwf1[i, 2]
        h_i = hwf1[i, 0]
        w_i = hwf1[i, 1]

        # x - W
        x = -focal_i * (pts_i_cam[:, 0] / pts_i_cam[:, 2])
        # y - H
        y = -focal_i * (pts_i_cam[:, 1] / pts_i_cam[:, 2])

        xyz = np.stack([x, y, pts_i_cam[:, 2]]).T

        visible_pts_map = (xyz[:, 2] > 0) & (np.abs(xyz[:, 0]) < w_i // 2) & (np.abs(xyz[:, 1]) < h_i // 2)

        xyz_visible = xyz[visible_pts_map]

        # xxx['coord'][:, 0] == W
        # xxx['coord'][:, 1] == H
        xyz_visible[:, 0] = np.maximum(np.minimum(xyz_visible[:, 0] + w_i // 2, w_i), 0)
        xyz_visible[:, 1] = np.maximum(np.minimum(xyz_visible[:, 1] + h_i // 2, h_i), 0)

        xyz_pts.append(
            {
                'depth': xyz_visible[:, 2],
                'coord': xyz_visible[:, :2],
                'weight': np.ones_like(xyz_visible[:, 2])
            }
        )

        # ######### Debug Depth Outputs
        # if i == 102:
        #     scale = 8
        #
        #     h_scaled = h_i // scale
        #     w_scaled = w_i // scale
        #
        #     xyz_vis_scaled = xyz_visible / 8
        #
        #     depth_img = np.zeros([int(h_scaled), int(w_scaled), 1])
        #     depth_img[np.floor(xyz_vis_scaled[:, 1]).astype("int"), np.floor(xyz_vis_scaled[:, 0]).astype("int")] = xyz_visible[:, 2][:, None]
        #     plt.figure()
        #     plt.imshow(depth_img[..., 0], cmap="plasma")
        #
        #     plt.figure()
        #     img_i = Image.open(img_path[i])
        #     img_i = img_i.resize((w_scaled, h_scaled))
        #     plt.imshow(np.asarray(img_i))
        #
        #     # pcd = o3d.geometry.PointCloud()
        #     # pcd.points = o3d.utility.Vector3dVector(pts_i_cam)

    np.save(os.path.join(segmemt_pth, pt_depth_file_n), xyz_pts)


    # plt.scatter(cam_pose_openGL[:, 0, 3], cam_pose_openGL[:, 1, 3])
    # plt.axis('equal')
    #
    # for i in range(len(pos)):
    #     p = cam_pose_openGL[i]
    #     # p = pos[i].dot(np.array([[-1., 0., 0., 0., ],
    #     #                          [0., 1., 0., 0., ],
    #     #                          [0., 0., -1., 0., ],
    #     #                          [0., 0., 0., 1., ], ]))
    #     plt.arrow(p[0, 3], p[1, 3], p[0, 0], p[1, 0], color="red")
    #     plt.arrow(p[0, 3], p[1, 3], p[0, 2], p[1, 2], color="black")



def extract_depth_information(dataset, exp_dict):
    pass


def extract_depth_image(dataset, exp_dict):
    pass