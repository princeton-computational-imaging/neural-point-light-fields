import numpy as np
import torch
from pytorch3d.transforms import Rotate, Transform3d
from src.scenes.frames import GraphEdge


def get_frustum_world(camera, edge2camera, device='cpu'):
    cam_trafo = edge2camera.get_transformation_c2p().to(device=device)
    H = camera.intrinsics.H
    W = camera.intrinsics.W
    focal = camera.intrinsics.f_x
    sensor_corner = torch.tensor([[0., 0.], [0., H - 1], [W - 1, H - 1], [W - 1, 0.]], device=device)
    frustum_edges = torch.stack([(sensor_corner[:, 0] - W * .5) / focal, -(sensor_corner[:, 1] - H * .5) / focal,
                                 -torch.ones(size=(4,), device=device)], -1)
    frustum_edges /= torch.norm(frustum_edges, dim=-1)[:, None]
    frustum_edges = \
        Rotate(camera.R, device=device).compose(cam_trafo.translate(-edge2camera.translation)).transform_points(
            frustum_edges.reshape(1, -1, 3))[0]
    frustum_edges /= torch.norm(frustum_edges, dim=-1)[:, None]

    frustum_normals = torch.cross(frustum_edges, frustum_edges[[1, 2, 3, 0], :])

    frustum_normals /= torch.norm(frustum_normals, dim=-1)[:, None]

    return frustum_edges, frustum_normals


def get_frustum(camera, edge2camera, edge2reference, device='cpu'):
    # Gets the edges and normals of a cameras frustum with respect to a reference system
    openGL2dataset = Rotate(camera.R, device=device)

    if type(edge2reference) == GraphEdge:
        ref2cam = torch.eye(4, device=device)[None]

        ref2cam[0, 3, :3] = edge2reference.translation - edge2camera.translation
        ref2cam[0, :3, :3] = edge2camera.getRotation_c2p().compose(edge2reference.getRotation_p2c()).get_matrix()[:, :3, :3]
        cam_trafo = Transform3d(matrix=ref2cam)

    else:
        wo2veh = edge2reference
        cam2wo = edge2camera.get_transformation_c2p().cpu().get_matrix()[0].T.detach().numpy()

        cam2veh = wo2veh.dot(cam2wo)
        cam2veh = Transform3d(matrix=torch.tensor(cam2veh.T, device=device, dtype=torch.float32))
        cam_trafo = cam2veh
        ref2cam = cam2veh.get_matrix()

    H = camera.intrinsics.H
    W = camera.intrinsics.W
    focal = camera.intrinsics.f_x.detach()
    sensor_corner = torch.tensor([[0., 0.], [0., H - 1], [W - 1, H - 1], [W - 1, 0.]], device=device)
    frustum_edges = torch.stack([(sensor_corner[:, 0] - W * .5) / focal, -(sensor_corner[:, 1] - H * .5) / focal,
                                 -torch.ones(size=(4,), device=device)], -1)
    frustum_edges /= torch.norm(frustum_edges, dim=-1)[:, None]
    frustum_edges = openGL2dataset.compose(cam_trafo.translate(-ref2cam[:, 3, :3])).transform_points(
        frustum_edges.reshape(1, -1, 3))[0]
    frustum_edges /= torch.norm(frustum_edges, dim=-1)[:, None]
    frustum_normals = torch.cross(frustum_edges, frustum_edges[[1, 2, 3, 0], :])
    frustum_normals /= torch.norm(frustum_normals, dim=-1)[:, None]
    return frustum_edges, frustum_normals


def get_frustum_torch(camera, edge2camera, edge2reference, device='cpu'):
    # Gets the edges and normals of a cameras frustum with respect to a reference system
    openGL2dataset = Rotate(camera.R, device=device)

    if type(edge2reference) == GraphEdge:
        ref2cam = torch.eye(4, device=device)[None]

        ref2cam[0, 3, :3] = edge2reference.translation - edge2camera.translation
        ref2cam[0, :3, :3] = edge2camera.getRotation_c2p().compose(edge2reference.getRotation_p2c()).get_matrix()[:, :3,
                             :3]
        cam_trafo = Transform3d(matrix=ref2cam)

    else:
        wo2veh = edge2reference
        cam2wo = edge2camera.get_transformation_c2p(device='cpu', requires_grad=False).get_matrix()[0].T

        cam2veh = torch.matmul(wo2veh, cam2wo)
        cam2veh = Transform3d(matrix=cam2veh.T)
        cam_trafo = cam2veh
        ref2cam = cam2veh.get_matrix()

    H = camera.intrinsics.H
    W = camera.intrinsics.W
    focal = camera.intrinsics.f_x.detach()
    sensor_corner = torch.tensor([[0., 0.], [0., H - 1], [W - 1, H - 1], [W - 1, 0.]], device=device)
    frustum_edges = torch.stack([(sensor_corner[:, 0] - W * .5) / focal, -(sensor_corner[:, 1] - H * .5) / focal,
                                 -torch.ones(size=(4,), device=device)], -1)
    frustum_edges /= torch.norm(frustum_edges, dim=-1)[:, None]
    frustum_edges = openGL2dataset.compose(cam_trafo.translate(-ref2cam[:, 3, :3])).transform_points(
        frustum_edges.reshape(1, -1, 3))[0]
    frustum_edges /= torch.norm(frustum_edges, dim=-1)[:, None]
    frustum_normals = torch.cross(frustum_edges, frustum_edges[[1, 2, 3, 0], :])
    frustum_normals /= torch.norm(frustum_normals, dim=-1)[:, None]
    return frustum_edges, frustum_normals


# TODO: Convert to full numpy version
def get_frustum_np(camera, edge2camera, edge2reference, device='cpu'):
    # Gets the edges and normals of a cameras frustum with respect to a reference system

    openGL2dataset = Rotate(camera.R, device=device)

    ref2cam = torch.eye(4, device=device)[None]

    ref2cam[0, 3, :3] = edge2reference.translation - edge2camera.translation
    ref2cam[0, :3, :3] = edge2camera.getRotation_c2p().compose(edge2reference.getRotation_p2c()).get_matrix()[:, :3, :3]
    cam_trafo = Transform3d(matrix=ref2cam)

    H = camera.intrinsics.H
    W = camera.intrinsics.W
    focal = camera.intrinsics.f_x
    sensor_corner = torch.tensor([[0., 0.], [0., H - 1], [W - 1, H - 1], [W - 1, 0.]], device=device)
    frustum_edges = torch.stack([(sensor_corner[:, 0] - W * .5) / focal, -(sensor_corner[:, 1] - H * .5) / focal,
                                 -torch.ones(size=(4,), device=device)], -1)
    frustum_edges /= torch.norm(frustum_edges, dim=-1)[:, None]
    frustum_edges = openGL2dataset.compose(cam_trafo.translate(-ref2cam[:, 3, :3])).transform_points(
        frustum_edges.reshape(1, -1, 3))[0]
    frustum_edges /= torch.norm(frustum_edges, dim=-1)[:, None]
    frustum_normals = torch.cross(frustum_edges, frustum_edges[[1, 2, 3, 0], :])
    frustum_normals /= torch.norm(frustum_normals, dim=-1)[:, None]
    return frustum_edges, frustum_normals