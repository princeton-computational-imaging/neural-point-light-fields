import numpy as np
import torch
from pytorch3d.transforms.so3 import so3_log_map, so3_exponential_map
from pytorch3d.transforms.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
)
from pytorch3d.transforms import Transform3d
from PIL import Image

from .. import utils as ut
import open3d as o3d

DEVICE = "cpu"
TRAINTRAFO = False
axis_4 = torch.tensor([[0.0, 0.0, 0.0, 1.0]])


# WAYMO
id2pos = {1: "FRONT",
          2: "FRONT_LEFT",
          3: "FRONT_RIGHT",}


class Frame:
    def __init__(self, root_node, image, camera, lidar, input_graph, nodes, point_cloud_pth, global_transformation=None):
        self.nodes = nodes
        self.images = {}
        self.wht_pt = {}
        self.alpha_contrast = {}
        self.beta_brightness = {}
        self.gamma = {}
        self.point_cloud_pth = {}
        self.background_embeddings = {}
        self.renderings = []
        self.scene_matrix = None

        # Scene Graph
        self.edges = {}
        self.camera_ids = list(camera.keys())

        self.global_transformation = global_transformation

        self.background_node = root_node.scene_idx

        if lidar is not None:
            for pcd_path, li in zip(point_cloud_pth, lidar.values()):
                self.point_cloud_pth[li.scene_idx] = pcd_path
                self.point_cloud = {}
                self.merged_pcd_pth = None
                self.merged_pcd = None

        # Add scene_graph
        edges = self.graphFromList(input_graph)

        self.add_edges(edges)

        for img, cam in zip(image, camera.values()):
            self.images[cam.scene_idx] = str(img)
            self.wht_pt[cam.scene_idx] = {'R': torch.tensor([1.], requires_grad=True),
                                          'G': torch.tensor([1.], requires_grad=False),
                                          'B': torch.tensor([1.], requires_grad=True),
                                          }

            self.alpha_contrast[cam.scene_idx] = torch.tensor([1.], requires_grad=True)
            self.beta_brightness[cam.scene_idx] = torch.tensor([0.], requires_grad=True)
            self.gamma[cam.scene_idx] = torch.tensor([.45], requires_grad=True)

        cam = self.nodes['camera'][cam.scene_idx]
        self.H = cam.H
        self.W = cam.W

    def load_image(self, camera_idx, adjust_for_front=False):
        assert (
            camera_idx in self.camera_ids
        ), f"only cameras {self.camera_ids} are available"
        cam = list(self.nodes["camera"].values())[0]
        H, W = int(cam.H), int(cam.W)
        img_pil = Image.open(self.images[camera_idx])
        img_pil = img_pil.resize((W, H))
        img = (np.maximum(np.minimum(np.array(img_pil), 255), 0) / 255.0).astype(
            np.float32
        )

        return img

    def load_tone_mapping(self, camea_idx):
        return {'wht_pt': self.wht_pt[camea_idx],
                'contrast': self.alpha_contrast[camea_idx],
                'brightness': self.beta_brightness[camea_idx],
                'gamma': self.gamma[camea_idx],
                }

    def load_point_cloud(self, lidar_idx, caching=True):
        # point_cloud = np.fromfile(self.point_cloud_pth, dtype=np.float32, count=-1).reshape([-1, 4])
        # return point_cloud
        # TODO: Remove caching after adding to get_item
        # if self.point_cloud is None:
        #     self.point_cloud = np.fromfile(self.point_cloud_pth, dtype=np.float32, count=-1).reshape([-1, 4])
        # # point_cloud = torch.tensor(point_cloud, dtype=torch.float32, requires_grad=False).view(-1, 4)
        # return self.point_cloud
        if caching and lidar_idx in self.point_cloud:
            points = self.point_cloud[lidar_idx]

        else:
            if self.point_cloud_pth[lidar_idx][-3:] == 'ply' or self.point_cloud_pth[lidar_idx][-3:] == 'pcd':
                points = o3d.io.read_point_cloud(self.point_cloud_pth[lidar_idx]).points
                points = np.asarray(points)
            else:
                points = np.fromfile(self.point_cloud_pth[lidar_idx], dtype=np.float32, count=-1).reshape([-1, 4])

            if caching:
                self.point_cloud[lidar_idx] = points

        return points

    def load_mask(self, camera_idx, return_xycfn=False, return_type_id=True):
        assert (
            camera_idx in self.camera_ids
        ), f"only cameras {self.camera_ids} are available"

        # mask = imageio.imread(self.images[camera_idx].replace("image_02", "instances"))
        mask_pil = Image.open(self.images[camera_idx].replace("image_02", "instances"))
        mask_pil = mask_pil.resize((self.W, self.H), resample=Image.NEAREST)
        mask = np.array(mask_pil)
        type_idx_list = {v.type_idx: k for k, v in self.nodes["scene_object"].items()}
        mask_new = np.zeros((self.H, self.W))
        for mask_id in np.unique(mask):
            if mask_id == 0:
                continue
            class_id = mask_id // 1000
            type_idx = mask_id % 1000
            if type_idx in type_idx_list:
                # assert type_idx != 0
                mask_new[mask == mask_id] = type_idx_list[type_idx]

        # hu.save_image('tmp.png', imageio.imread(self.images[camera_idx]), mask=mask_new)

        if return_xycfn:
            xycfn = ut.mask_to_xycfn(self.frame_idx, camera_idx, mask_new)
            return xycfn
        if return_type_id:
            return mask_new
        return mask

    def add_edges(self, edges):
        if len(self.edges) > 0:
            last_ed_id = max(self.edges.keys()) + 1
        else:
            last_ed_id = 0

        scene_edge_list = []
        for i, e in enumerate(edges):
            e.index = i + last_ed_id
            self.edges[i + last_ed_id] = e
            p = torch.as_tensor(e.parent)[None]
            c = torch.as_tensor(e.child)[None]
            scene_edge_list += [
                torch.cat([p, c, e._transformation.cpu().detach().flatten()], dim=0)
            ]

        if self.scene_matrix is None:
            self.scene_matrix = torch.stack(scene_edge_list, dim=0)
        else:
            self.scene_matrix = torch.cat(
                [self.scene_matrix, torch.stack(scene_edge_list, dim=0)]
            )

    def set_rotation(self, node_id, rotation):
        # get rotation
        edges_bck2obj = self.get_all_edges_to_node_by_id(node_id)
        edge_id = edges_bck2obj[0].index
        self.edges[edge_id].rotation.data[:] = rotation

    def get_object_parameters(self, node_id):
        node = self.nodes["scene_object"][node_id]
        # get latent
        if not hasattr(node, "latent"):
            latent = torch.zeros(256).cuda()
        else:
            latent = self.nodes["scene_object"][node_id].latent

        # get scaling
        edges_bck2obj = self.get_edge_by_parent_idx([node_id])
        edge_id = edges_bck2obj[0].index
        scaling = self.edges[edge_id].scaling

        # get rotation
        edges_bck2obj = self.get_all_edges_to_node_by_id(node_id)
        edge_id = edges_bck2obj[0].index
        tran = self.edges[edge_id].translation
        rot = self.edges[edge_id].rotation

        return {
            "node": self.nodes["scene_object"][node_id],
            "latent": latent,
            "scaling": scaling,
            "translation": tran,
            "rotation": rot,
            "cam_translation": self.edges[0].translation,
            "cam_rotation": self.edges[0].rotation,
            "node_idx": node_id,
        }

    def get_objects_ids(self):
        return [
            int(s[1])
            for s in self.scene_matrix
            if int(s[1]) in self.nodes["scene_object"]
        ]

    def get_object_params(self):
        return [self.get_object_parameters(o) for o in self.get_objects_ids()]

    def get_objects(self):
        return [self.nodes[o] for o in self.get_objects_ids()]

    def remove_object_by_id(self, node_id):
        edges_bck2obj = self.get_all_edges_to_node_by_id(node_id)
        if len(edges_bck2obj) == 0:
            print(f"{node_id} does not exist")
            return

        edges_obj2class = self.get_edge_by_parent_idx([node_id])
        for edge in edges_bck2obj + edges_obj2class:
            self.removeEdge(edge.index)
            # print(edge, 'removed')

    # def addPointCloud(self, point_cloud):
    #     # No checks necessary for now
    #     self.point_cloud = torch.tensor(point_cloud, dtype=torch.float32, requires_grad=False).view(-1, 4)

    def graphFromList(self, graph_list):
        """
        List of parent_node, child_node, transformation
        List of type [n_edges, parent+child+transformation]
        """
        edge_list = []
        for edge in graph_list:
            parent = edge[0]
            child = edge[1]
            transformation = None
            if len(edge) == 3:
                transformation = edge[2]

            edge_list += [GraphEdge(parent, child, transformation)]
        return edge_list

    def graphFromArray(self, graph_array):
        """
        Array, witth id of parent_nodes in [0], child_nodes in [1], and transformation in [2:]
        [n_edges, 2 + transformation.len()]
        """
        for edge in graph_array:
            self.addEdge(edge[0], edge[1], edge[2:])

    # def addEdge(self, parent_node, child_node, transformation=None,):
    #     # TODO: Try if parent node and child node are part of this scenes graph and show error if not
    #     new_edge = GraphEdge(parent_node, child_node, transformation, edge_id=len(self.edges))
    #     self.edges[new_edge.index] = new_edge
    #     self.updateSceneMatrix(new_edge)

    def removeEdge(self, edge_id):
        self.scene_matrix[edge_id][:] = -1

        del self.edges[edge_id]

    def get_edges(self):
        return [k for k in self.scene_matrix[:, :2]]

    def get_all_edges_to_node_by_id(self, node_scene_idx):
        idx = np.where(self.scene_matrix[:, 1] == node_scene_idx)
        return [self.edges[i] for i in idx[0]]

    def get_edge_by_idx(self, idx):
        return [self.edges[i] for i in idx]

    def get_edge_by_child_idx(self, child_idx):
        edge_idx = self.get_edge_idx_by_child_idx(child_idx)
        return [self.get_edge_by_idx(idx) for idx in edge_idx]

    def get_edge_by_parent_idx(self, parent_idx):
        edge_idx = self.get_edge_idx_by_parent_idx(parent_idx)
        return self.get_edge_by_idx(edge_idx)

    def get_edge_idx_by_child_idx(self, child_idx):
        edge_idx = []
        matrix = self.scene_matrix
        for node_idx in child_idx:
            idx = list(np.where(matrix[:, 1] == node_idx)[0].astype(int))
            edge_idx.append(idx)
        return edge_idx

    def get_edge_idx_by_parent_idx(self, parent_nodes):
        edge_idx = np.array([])
        matrix = self.scene_matrix
        for node_idx in parent_nodes:
            edge_idx = np.concatenate([edge_idx, np.where(matrix[:, 0] == node_idx)[0]])
        return list(edge_idx.astype(int))

    def get_edges_to_root_by_idx(self, child_idx: list, edges: list = []):
        # TODO: Finish to generalize
        edges = []
        checked = []
        while child_idx:
            checked += child_idx
            new_child_idx = []
            for idx in child_idx:
                connecting_edges = self.get_edge_by_child_idx([idx])
                for e in connecting_edges:
                    if e.parent not in checked and e.parent != self.root_node:
                        new_child_idx.append(e.parent)
            child_idx = new_child_idx

        return checked

    def get_edge_transformations(self, edge_idx):

        rotations = torch.cat([self.edges[ed_i].rotation for ed_i in edge_idx])
        translations = torch.cat([self.edges[ed_i].translation for ed_i in edge_idx])

        rot_mat = so3_exponential_map(rotations)

        trafo_mat = torch.cat([rot_mat, translations[:, :, None]], dim=2)

        ax = axis_4.to(trafo_mat.device).repeat(len(trafo_mat), 1)[:, None]

        return Transform3d(
            matrix=torch.cat([trafo_mat, ax], dim=1).transpose(2, 1)
        ).inverse()

    def get_edge_scalings(self, edge_idx):
        scale = torch.cat([self.edges[ed_i].scaling[None] for ed_i in edge_idx])

        trafo_mat = (
            torch.eye(4)[None]
            * torch.cat([scale, torch.full([len(scale), 1], 1.0)], dim=1)[:, None]
        )
        return Transform3d(matrix=trafo_mat)


class GraphEdge:
    """
    translation: describes the translation from the parent to
    the child node in the reference system of the parent node
    rotation: Describes the axis of the child node in the
    reference system of the parent node
    """

    def __init__(self, parent_node, child_node, transformation=None):
        self.parent = int(parent_node)
        self.child = int(child_node)
        self._transformation = torch.eye(4).to(DEVICE)
        self.translation = self._transformation[None, :3, 3]
        self.delta_translation = torch.tensor([[0., 0., 0., ]], requires_grad=True, )
        self.rotation = torch.zeros([1, 3]).to(DEVICE)
        self.delta_rotation = torch.tensor([[0., 0., 0., ]], requires_grad=True, )
        self.scaling = torch.ones([3]).to(DEVICE)

        self._transformation3D = None
        self._translation3D = None
        self._rotation3D = None
        self._scaling3D = None

        if transformation is not None:
            self._updateTransformation(transformation)

    def update(self, parent_node=None, child_node=None, transformation=None):
        if parent_node is not None:
            self.parent = parent_node

        if child_node is not None:
            self.child = child_node

        if transformation is not None:
            self._updateTransformation(transformation)

    def _updateTransformation(self, transformation):
        self._checkTrafoValidity(transformation)
        if isinstance(transformation, np.ndarray):
            if transformation.dtype == "float64":
                transformation = np.float32(transformation)
            transformation = torch.from_numpy(transformation)

        affine_trafo = self._affineTrafo(transformation).to(self.translation.device)
        self.translation.data = (
            affine_trafo[None, :3, 3].clone().detach().requires_grad_(TRAINTRAFO)
        )
        self.translation.requires_grad_(TRAINTRAFO)

        # If a rotation matrix get skew
        if not affine_trafo[:3, :3].to(torch.float64).diagonal().norm() == affine_trafo[
            :3, :3
        ].to(torch.float64).norm() or any(affine_trafo[:3, :3].diagonal() < 0.0):
            # Check if rotation is orthogonal
            if torch.norm(torch.det(affine_trafo[:3, :3])) != 1.:
                affine_trafo[:3, :3] = \
                euler_angles_to_matrix(matrix_to_euler_angles(affine_trafo[None, :3, :3], 'XYZ'), 'XYZ')[0]

            rot_angles = so3_log_map(affine_trafo[None, :3, :3].clone().detach())
            rot_angles = torch.tensor([[r if torch.absolute(r) > 1.e-6 else 0. for r in rot_angles.squeeze()]], device=rot_angles.device)
            self.rotation = rot_angles
            self.rotation.requires_grad = TRAINTRAFO
        else:
            # If a scaling matrix only set diagonal
            # TODO: Make sure gradient flows back to node dimension and does not get stuck at scaling
            self.scaling.data = (
                affine_trafo[:3, :3].diagonal().detach().requires_grad_(TRAINTRAFO)
            )
            self.scaling.requires_grad_(TRAINTRAFO)

        self._initial_transformation = affine_trafo

    def _checkTrafoValidity(self, transformation):
        assert isinstance(
            transformation, (np.ndarray, torch.Tensor)
        ), "Transformation is of the wrong type. Must be numpy array or torch tensor."
        assert (
            transformation.flatten().shape[0] <= 16
        ), "Transformation must be an affine 3D transformation"

    def _affineTrafo(self, transformation):
        transformation = transformation.to(DEVICE)
        affine = torch.eye(4).to(DEVICE)
        length = transformation.flatten().shape[0]

        # translation
        if length == 3:
            affine[:3, 3] = transformation.reshape([3])
        # rotation
        elif length == 4:
            A = euler_angles_to_matrix(
                torch.Tensor([0.0, transformation[3], 0.0]), "ZYX"
            )
            affine[:3, :3] = A
            affine[:3, 3] = transformation[:3].reshape([3])

        elif length == 9:
            affine[:3, :3] = transformation.reshape([3, 3])

        elif length == 12:
            affine[:3, :] = transformation.reshape([3, 4])

        elif length == 16:
            affine = transformation.reshape([4, 4])

        return affine

    def _initTrafoSE3(self, mean=0.0, std=1e-6):
        """
        Initalize Trafo as special Euclidean group SE(3), with
        translation t in R3 and Rotation R as special orthogonal
         group SO(3).
        t_x, t_y, t_z, w_0, w_1, w_2
        """
        self.translation = torch.normal(mean=mean, std=std, size=(3,))
        self.rotation = torch.normal(mean=mean, std=std, size=(3,))

    def _getTransformation(
        self, rotation_only=False, translation_only=False, device=None, requires_grad=True,
    ):
        if device is None:
            device = self.translation.device

        if requires_grad:
            translation = self.translation.to(device) + self.delta_translation.to(device)
            rotation = self.rotation.to(device) + self.delta_rotation.to(device)
        else:
            translation = self.translation.detach().to(device)
            rotation = self.rotation.detach().to(device)

        rot_mat = self.so3_exponential_map(rotation)
        if rotation_only:
            trafo_mat = torch.cat(
                [rot_mat, torch.zeros(3, device=device)[None, :, None]], dim=2
            )
        elif translation_only:
            trafo_mat = torch.cat(
                [
                    torch.zeros([3, 3], device=device)[None],
                    translation[..., None],
                ],
                dim=2,
            )
        else:
            trafo_mat = torch.cat([rot_mat, translation[..., None]], dim=2)

        trafo_mat = torch.cat(
            [trafo_mat, axis_4.to(trafo_mat.device)[:, None]], dim=1
        ).transpose(2, 1)
        return Transform3d(matrix=trafo_mat)

    def get_transformation_p2c(self):
        return self._getTransformation().inverse()

    def get_transformation_c2p(self, **kwargs):
        return self._getTransformation(**kwargs)

    def getTransformation(self):
        return self.get_transformation_p2c()

    def getRotation_c2p(self):
        return self._getTransformation(rotation_only=True)

    def getRotation_p2c(self):
        return self._getTransformation(rotation_only=True).inverse()

    def getRotation(self):
        return self.getRotation_p2c()

    def getTranslation_c2p(self):
        return self._getTransformation(translation_only=True)

    def getTranslation_p2c(self):
        return self._getTransformation(translation_only=True).inverse()

    def getTranslation(self):
        return self.getTranslation_p2c()

    def getScaling(self):
        scaling = torch.cat([self.scaling, torch.ones(1, device=DEVICE)])
        trafo_mat = torch.diag(scaling)[None, ...]
        return Transform3d(matrix=trafo_mat)

    def getEuler(self):
        return matrix_to_euler_angles(self.transformation[:3, :3])

    # Copy to make this method faster
    def so3_exponential_map(self, log_rot, eps: float = 0.0001):
        """
        Convert a batch of logarithmic representations of rotation matrices `log_rot`
        to a batch of 3x3 rotation matrices using Rodrigues formula [1].

        In the logarithmic representation, each rotation matrix is represented as
        a 3-dimensional vector (`log_rot`) who's l2-norm and direction correspond
        to the magnitude of the rotation angle and the axis of rotation respectively.

        The conversion has a singularity around `log(R) = 0`
        which is handled by clamping controlled with the `eps` argument.

        Args:
            log_rot: Batch of vectors of shape `(minibatch , 3)`.
            eps: A float constant handling the conversion singularity.

        Returns:
            Batch of rotation matrices of shape `(minibatch , 3 , 3)`.

        Raises:
            ValueError if `log_rot` is of incorrect shape.

        [1] https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        """

        _, dim = log_rot.shape
        if dim != 3:
            raise ValueError("Input tensor shape has to be Nx3.")

        nrms = (log_rot * log_rot).sum(1)
        # phis ... rotation angles
        rot_angles = torch.clamp(nrms, eps).sqrt()
        rot_angles_inv = torch.reciprocal(rot_angles)
        fac1 = rot_angles_inv * rot_angles.sin()
        fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
        skews = self.hat(log_rot)  # torch.rand(1,3,3).to(fac1.device) #

        R = (
            # pyre-fixme[16]: `float` has no attribute `__getitem__`.
            fac1[:, None, None] * skews
            + fac2[:, None, None] * torch.bmm(skews, skews)
            + torch.eye(3, dtype=log_rot.dtype, device=DEVICE)[None]
        )

        return R

    def hat(self, v):
        """
        Compute the Hat operator [1] of a batch of 3D vectors.

        Args:
            v: Batch of vectors of shape `(minibatch , 3)`.

        Returns:
            Batch of skew-symmetric matrices of shape
            `(minibatch, 3 , 3)` where each matrix is of the form:
                `[    0  -v_z   v_y ]
                 [  v_z     0  -v_x ]
                 [ -v_y   v_x     0 ]`

        Raises:
            ValueError if `v` is of incorrect shape.

        [1] https://en.wikipedia.org/wiki/Hat_operator
        """

        N, dim = v.shape
        if dim != 3:
            raise ValueError("Input vectors have to be 3-dimensional.")

        h = v.new_zeros(N, 3, 3)

        x, y, z = v.unbind(1)

        h[:, 0, 1] = -z
        h[:, 0, 2] = y
        h[:, 1, 0] = z
        h[:, 1, 2] = -x
        h[:, 2, 0] = -y
        h[:, 2, 1] = x

        return h
