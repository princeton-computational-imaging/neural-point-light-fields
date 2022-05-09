import numpy as np
import torch
import torch.nn as nn
from pytorch3d.renderer import PerspectiveCameras
from src.pointLF.point_light_field import PointLightField
from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix

DEVICE = "cpu"
TRAINCAM = False
TRAINOBJSZ = False
TRAINOBJ = False


class Intrinsics:
    def __init__(
        self,
        H,
        W,
        focal,
        P=None
    ):
        self.H = int(H)
        self.W = int(W)
        if np.size(focal) == 1:
            self.f_x = nn.Parameter(
                torch.tensor(focal, device=DEVICE, requires_grad=TRAINCAM)
            )
            self.f_y = nn.Parameter(
                torch.tensor(focal, device=DEVICE, requires_grad=TRAINCAM)
            )
        else:
            self.f_x = nn.Parameter(
                torch.tensor(focal[0], device=DEVICE, requires_grad=TRAINCAM)
            )
            self.f_y = nn.Parameter(
                torch.tensor(focal[1], device=DEVICE, requires_grad=TRAINCAM)
            )

        self.P = P


class NeuralCamera(PerspectiveCameras):
    def __init__(self, h, w, f, intrinsics=None, name=None, type=None, P=None):
        # TODO: Cleanup intrinsics and h, w, f
        # TODO: Add P matrix for projection
        self.H = h
        self.W = w
        if intrinsics is None:
            self.intrinsics = Intrinsics(h, w, f, P)
        else:
            self.intrinsics = intrinsics

        # Add opengl2cam rotation
        opengl2cam = euler_angles_to_matrix(
            torch.tensor([np.pi, np.pi, 0.0], device=DEVICE), "ZYX"
        )
        if type == 'waymo':
            waymo_rot = euler_angles_to_matrix(torch.tensor([np.pi, 0., 0.], 
             device=DEVICE), 'ZYX')

            opengl2cam = torch.matmul(waymo_rot, opengl2cam)

        # Simplified version under the assumption of square pixels
        #  [self.intrinsics.f_x, self.intrinsics.f_y]
        PerspectiveCameras.__init__(
            self,
            focal_length=torch.tensor([[self.intrinsics.f_x, self.intrinsics.f_y]]),
            principal_point=torch.tensor(
                [[self.intrinsics.W / 2, self.intrinsics.H / 2]]
            ),
            R=opengl2cam,
            image_size=torch.tensor([[self.intrinsics.W, self.intrinsics.H]]),
        )
        self.name = name


class Lidar:
    def __init__(self, Tr_li2cam=None, name=None):
        # TODO: Add all relevant params to scene init
        self.sensor_type = 'lidar'
        self.li2cam = Tr_li2cam

        self.name = name


class ObjectClass:
    def __init__(self, name):
        self.static = False
        self.name = name


class SceneObject:
    def __init__(self, length, height, width, object_class_node):
        self.static = False
        self.object_class_type_idx = object_class_node.type_idx
        self.object_class_name = object_class_node.name
        self.length = length
        self.height = height
        self.width = width
        self.box_size = torch.tensor([self.length, self.height, self.width])


class Background:
    def __init__(self, transformation=None, near=0.5, far=100.0, lightfield_config={}):
        self.static = True
        global_transformation = np.eye(4)

        if transformation is not None:
            transformation = np.squeeze(transformation)
            if transformation.shape == (3, 3):
                global_transformation[:3, :3] = transformation
            elif transformation.shape == (3):
                global_transformation[:3, 3] = transformation
            elif transformation.shape == (3, 4):
                global_transformation[:3, :] = transformation
            elif transformation.shape == (4, 4):
                global_transformation = transformation
            else:
                print(
                    "Ignoring wolrd transformation, not of shape [3, 3], [3, 4], [3, 1] or [4, 4], but",
                    transformation.shape,
                )

        self.transformation = torch.from_numpy(global_transformation)
        self.near = torch.tensor(near)
        self.far = torch.tensor(far)