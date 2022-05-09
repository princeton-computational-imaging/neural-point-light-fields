import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.manifold import TSNE


def plt_pts_selected_2D(output_dict, axis):
    selected_points = [pts[mask] for pts, mask in
                       zip(output_dict['points_in'], output_dict['closest_mask_in'])]

    for l, cf in enumerate(output_dict['samples']):
        plt_pts = selected_points[l].reshape(-1, 3)
        all_pts = output_dict['points_in'][l]

        plt.figure()
        plt.scatter(all_pts[:, axis[0]], all_pts[:, axis[1]])
        plt.scatter(plt_pts[:, axis[0]], plt_pts[:, axis[1]])

    if 'points_scaled' in output_dict:
        selected_points = [pts[mask] for pts, mask in
                           zip(output_dict['points_scaled'], output_dict['closest_mask_in'])]
        for l, cf in enumerate(output_dict['samples']):
            plt_pts = selected_points[l].reshape(-1, 3)
            all_pts = output_dict['points_scaled'][l]

            plt.figure()
            plt.scatter(all_pts[:, axis[0]], all_pts[:, axis[1]])
            plt.scatter(plt_pts[:, axis[0]], plt_pts[:, axis[1]])

def plt_BEV_pts_selected(output_dict):
    plt_pts_selected_2D(output_dict, (0,1))

def plt_SIDE_pts_selected(output_dict):
    plt_pts_selected_2D(output_dict, (0, 2))

def plt_FRONT_pts_selected(output_dict):
    plt_pts_selected_2D(output_dict, (1, 2))

def visualize_output(output_dict, selected_only=False, scaled=False, n_plt_rays=None):
    if scaled:
        pts_in = output_dict['points_scaled']
    else:
        pts_in = output_dict['points_in']

    masks_in = output_dict['closest_mask_in']

    if 'sum_mv_point_features' in output_dict:
        feat_per_point = output_dict['sum_mv_point_features'].squeeze()
        if len(feat_per_point.shape) == 4:
            n_batch, n_rays, n_closest_pts, feat_dim = feat_per_point.shape
        else:
            n_batch = 1
            n_rays, n_closest_pts, feat_dim = feat_per_point.shape

        for i, (pts, mask, feat) in enumerate(zip(pts_in, masks_in, feat_per_point)):

            # Get feature embedding for visualization
            feat = feat.reshape(-1, feat_dim)
            feat_embedded = TSNE(n_components=3).fit_transform(feat)

            # Transform embedded space to RGB
            feat_embedded = feat_embedded - feat_embedded.min(axis=0)
            color = feat_embedded / feat_embedded.max(axis=0)

            if n_plt_rays is not None:
                ray_ids = np.random.choice(len(mask), n_plt_rays)
                mask = mask[ray_ids]
                color = color.reshape(n_rays, n_closest_pts, 3)
                color = color[ray_ids].reshape(-1, 3)

            pts_close = pts[mask]
            pcd_close = get_pcd_vis(pts_close, color_vector=color)
            if selected_only:
                o3d.visualization.draw_geometries([pcd_close])
            else:
                pcd = get_pcd_vis(pts)
                o3d.visualization.draw_geometries([pcd, pcd_close])
    else:
        for i, (pts, mask) in enumerate(zip(pts_in, masks_in)):
            pts_close = pts[mask]
            pcd_close = get_pcd_vis(pts, uniform_color=[1., 0.7, 0.])

            if selected_only:
                o3d.visualization.draw_geometries([pcd_close])
            else:
                pcd = get_pcd_vis(pts)
                o3d.visualization.draw_geometries([pcd, pcd_close])

def get_pcd_vis(pts, uniform_color=None, color_vector=None):
    pts = pts.reshape(-1,3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if uniform_color is not None:
        # pcd.paint_uniform_color([1., 0.7, 0.])
        pcd.paint_uniform_color(uniform_color)
    if color_vector is not None:
        assert len(pts) == len(color_vector)
        pcd.colors = o3d.utility.Vector3dVector(color_vector)

    return pcd



# pts_in_name = ".tmp/points_in_{}.ply".format(i)
# o3d.io.write_point_cloud(pts_in_name, pcd)
# pcd = o3d.io.read_point_cloud(pts_in_name)