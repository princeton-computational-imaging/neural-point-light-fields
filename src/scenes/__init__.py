import os
import copy
import matplotlib.pyplot as plt
from typing import List, Dict
import tqdm
import pandas as pd
import numpy as np
import torch
from concurrent import futures
from multiprocessing import Pool, freeze_support, cpu_count

from .raysampler import NeuralSceneRaysampler, PointLightFieldSampler
from .. import datasets
from src.scenes.nodes import NeuralCamera, Lidar, SceneObject, ObjectClass, Background
from src.scenes.frames import Frame
from src.datasets.extract_baselines import extract_waymo_poses
from pytorch3d.transforms.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
)
from pytorch3d.transforms.so3 import so3_log_map, so3_exponential_map
from pytorch3d.transforms import Transform3d, Translate, Rotate, Scale
from .. import utils as ut
import time

_obj_class_label2sem = {0: "Car", 1: "Van", 2: "Truck", 3: "Tram", 4: "Pedestrian"}

axis_4 = torch.tensor([[0.0, 0.0, 0.0, 1.0]])


class NeuralScene:
    def __init__(self, scene_list, datadir, args, n_frames=None, exp_dict=None):
        """
        Data structure to store dynamic multi-object 3D scenes.
        Create a scene graph for each frame from tracking data and respective camera paramters.
        To use a scene for training ground truth images need to be added to each frame.

        Args:
            dataset:
            models:
            global_transformation:
        """
        self.device = "cpu"
        self.obj_only = exp_dict.get('obj_only', False)
        self.type2class = {}
        self.frames = {}
        self.nodes = {}
        self.frames_cameras = []
        self.scene_frames = {}

        # create counters
        self.scene_idx_counter = 0
        self.frame_idx_counter = 0
        self.meta = []

        lightfield_config = exp_dict.get('lightfield', None)

        # Set optimization behavior of neural scene
        self.tonemapping = False
        if lightfield_config.get("all_cams", False):
            self.tonemapping = exp_dict.get('tonemapping', False)
        self.refine_camera_pose = exp_dict.get("pose_refinement", False)
        self.recalibrate = False

        # TODO: Change behavior if training on multiple scenes
        self.scene_descriptor = scene_list[0]
        veh_world = True

        if not isinstance(scene_list, list):
            scene_list = [scene_list]

        for scene_dict in scene_list:
            # Preapre dataset for neural scene conversion
            dataset = datasets.get_dataset(datadir, scene_dict, args)
            self.dataset = dataset
            scale = exp_dict.get("scale", 1.0)
            print("Extracting Data for Baseline Experiments")

            # For the datasets with just a single camera sensor type (KITTI)
            if not isinstance(dataset.H, dict):
                def param2dict(par):
                    return {'LEFT': par, 'RIGHT': par}
                dataset.focal = param2dict(focal)
                dataset.H = param2dict(H)
                dataset.W = param2dict(W)

            self.H = {k: int(v * scale) for k, v in dataset.H.items()}
            self.W = {k: int(v * scale) for k, v in dataset.W.items()}
            self.focal = {k: float(v * scale) for k, v in dataset.focal.items()}
            self.hwf = {'width': self.W, 'height': self.H,'focal': self.focal}


            # Get and Update nodes
            s_time = time.time()
            node_list = nodes_from_dataset(
                dataset, neural_scene=self, object_class_list=scene_dict.get("object_class_list"), lightfield_config=lightfield_config
            )
            self.lightfield_config = lightfield_config if lightfield_config is not None else {}
            nodes_scene = self.updateNodes(node_list)
            print(f"Loaded Nodes in {time.time()-s_time:.3f} seconds")

            # get frames
            s_time = time.time()

            frame_list = frames_from_dataset(
                dataset, nodes=nodes_scene, n_frames=n_frames, veh_world=veh_world,
                all_cams= lightfield_config.get("all_cams", False),
                type=scene_list[0]['type'],
            )
            print(f"Loaded Frames in {time.time()-s_time:.3f} seconds")

            # Store meta data for each frame
            if type(scene_dict['scene_id']) == list:
                scene_dict['scene_id'] = tuple(scene_dict['scene_id'])
            self.scene_frames[scene_dict["scene_id"]] = []

            # Update self.frames in place
            self.updateFrames(
                frame_list, scene_id=scene_dict["scene_id"], scene_dict=scene_dict
            )

            self.scene_frames[scene_dict["scene_id"]] = np.array(
                self.scene_frames[scene_dict["scene_id"]]
            )
            if "object_class" in self.nodes:
                self.type2class = {
                    n.type_idx: n for n in self.nodes["object_class"].values()
                }
            else:
                # No objects e.g. single Light Field Rendering
                self.nodes["scene_object"] = {}

        self.frame_ids = list(self.frames.keys())

        self.raysampler = PointLightFieldSampler(
            scene=self,
            lightfield_config=lightfield_config,
            n_rays_per_image=exp_dict["n_rays"] // exp_dict["image_batch_size"],
            reference_frame='lidar',
            point_chunk_size=exp_dict.get('point_chunk', 1e12),
            exp_dict=exp_dict
        )

        print("Running Preprocessing of point cloud data. Merging data from multiple time steps.")
        if lightfield_config.get('merged_pcd', True) and lightfield_config.get('num_merged_frames', 2) > 1:
            self.init_neural_scene_data()

        self.ignore_list = set()
        # self.ignore_list.add(122)
        # self.ignore_list.add(123)

        # self.load_all_models(models)
        # filter out frame_cameras that don't have cars

        # Set validation and training split
        self.i_all = np.linspace(0, len(self.frames_cameras) - 1, len(self.frames_cameras), dtype=int)
        self.i_test = self.i_all[::np.ceil(1 / 0.1).astype(int)][1:]
        self.i_train = np.array([j for j in self.i_all if j not in self.i_test])

    def init_neural_scene_data(self):
        # Init all merged point clouds
        print("Joining and aligning Point Clouds across frames")
        s_time = time.time()
        pbar = tqdm.tqdm([fr_id for fr_id in self.frames.keys()], desc="Joining Point Clouds", leave=False)
        for fr_id in pbar:
            pts = self.raysampler._local_sampler.icp.forward(scene=self, cam_frame_id=fr_id, caching=False, augment=False)
            # Visualize for debugging
            # import open3d as o3d
            # open3d.visualization.draw
        print(f"Loaded Point Clouds in {time.time() - s_time:.3f} seconds")

    def wrapped_point_register(self, args):
        """
        we need to wrap the call to unpack the parameters
        we build before as a tuple for being able to use pool.map
        """
        self.raysampler._local_sampler.icp.forward(*args)

    def __len__(self):
        return len(self.frames_cameras)

    def __getitem__(self, idx, intersections_only=True, random_rays=True, manipulate=None, validation=False, EPI=False,
                    epi_row=None):
        # Optional learning of a scene calibration
        class null_with:
            def __enter__(self):
                pass

            def __exit__(self, a, b, c):
                pass

        optional_no_grad = null_with if (self.refine_camera_pose and self.recalibrate) else torch.no_grad

        # TODO: Get stats and update self.frames_cameras
        # idx = 122

        if idx in self.ignore_list:
            print("Ignoring Frame")
            return self.__getitem__(
                    np.random.choice(
                        [i for i in range(len(self)) if i not in self.ignore_list]
                    ),
                    intersections_only=intersections_only,
                random_rays=random_rays,
                )
        if idx in self.i_test and not validation:
            return self.__getitem__(
                np.random.choice(
                    [i for i in range(len(self)) if i not in self.i_test]
                ),
                intersections_only=intersections_only,
                random_rays=random_rays,
                validation=validation,
            )


        frame_idx, camera_idx, meta = self.frames_cameras[idx]
        frame = self.frames[frame_idx]

        # with torch.no_grad():
        with optional_no_grad():
            ray_dict = self.raysampler.forward(
                scene=self,
                frame_idx=frame_idx,
                camera_idx=camera_idx,
                intersections_only=intersections_only,
                random_rays=random_rays,
                obj_only=self.obj_only,
                validation=validation,
                EPI=EPI,
                epi_row=epi_row,
            )
            if ray_dict is None:
                self.ignore_list.add(idx)
                return self.__getitem__(
                    np.random.choice(
                        [i for i in range(len(self)) if i not in self.ignore_list]
                    ),
                    intersections_only=intersections_only,
                )

            images = torch.from_numpy(frame.load_image(camera_idx)).float()

            H, W, _ = images.shape
            xycfn = ray_dict["ray_bundle"].xys

        batch = {
            "frame_id": int(frame_idx),
            "camera_id": int(camera_idx),
            "meta": meta,
            "images": images,
            
            # "input_dict": input_dict,
            "H": H,
            "W": W,
        }

        batch.update(ray_dict)

        return batch

    def updateNodes(self, node_list):
        nodes_scene = {}
        for n in node_list:
            # Scene Idx
            if n["node_type"] == "object_class" and n["type_idx"] in self.type2class:
                scene_idx = self.type2class[n["type_idx"]].scene_idx
                n = self.nodes[scene_idx]

            else:
                scene_idx = self.scene_idx_counter
                self.scene_idx_counter += 1
            if not n["node_type"] in nodes_scene:
                nodes_scene[n["node_type"]] = {}
            if not n["node_type"] in self.nodes:
                self.nodes[n["node_type"]] = {}

            nodes_scene[scene_idx] = n
            self.nodes[scene_idx] = n

            nodes_scene[n["node_type"]][scene_idx] = n["node"]
            self.nodes[n["node_type"]][scene_idx] = n["node"]

            n["node"].scene_idx = scene_idx

        return nodes_scene

    def updateFrames(self, frame_list, scene_id, scene_dict):
        # Store frames in different dicts for better access
        for f in frame_list:
            # Store frames by idx
            self.frames[self.frame_idx_counter] = f
            self.frames[self.frame_idx_counter].frame_idx = self.frame_idx_counter
            for c in f.camera_ids:
                # Access specific images by frame and camera
                self.scene_frames[scene_id] += [len(self.frames_cameras)]
                self.frames_cameras += [(self.frame_idx_counter, c, scene_dict)]

            self.frame_idx_counter += 1

    def get_all_scene_graphs_matrix(self):
        graphs = []
        for frame_idx, frame in self.frames.items():
            graph_matrix = frame.scene_matrix[:, :2].astype(int)

            n_edges = graph_matrix.shape[0]
            rep_frame_idx = np.repeat(np.array([frame_idx]), n_edges)[:, None]
            edge_idx = np.linspace(0, n_edges - 1, n_edges, dtype=int)[:, None]

            graphs.append(
                np.concatenate([rep_frame_idx, edge_idx, graph_matrix], axis=1)
            )

        return np.array(graphs)

    def get_all_edges_to_cameras(self, camera_idx: list, frame_idx: list = None):
        if frame_idx is None:
            frame_idx = []
            cam_ids = camera_idx
            camera_idx = []
            for i in range(len(cam_ids)):
                frame_idx += list(self.frames.keys())
                camera_idx += [cam_ids[i]] * len(self.frames)

        edges2cams = {}
        for cam_i, frame_i in zip(camera_idx, frame_idx):
            if cam_i not in edges2cams:
                edges2cams[cam_i] = {}

            edges2cams[cam_i][frame_i] = self.frames[
                frame_i
            ].get_all_edges_to_node_by_id(self.nodes["camera"][cam_i].scene_idx)

        return edges2cams

    def get_all_edge_transformations(self, frame_edge_list):
        """
        List of tuples (frames_id, edge_id)
        """

        rotations = torch.cat(
            [self.frames[idx[0]].edges[idx[1]].rotation for idx in frame_edge_list]
        )
        translations = torch.cat(
            [self.frames[idx[0]].edges[idx[1]].translation for idx in frame_edge_list]
        )

        rot_mat = so3_exponential_map(rotations)

        trafo_mat = torch.cat([rot_mat, translations[:, :, None]], dim=2)

        ax = axis_4.to(self.device).repeat(len(trafo_mat), 1)[:, None]

        return Transform3d(
            matrix=torch.cat([trafo_mat, ax], dim=1).transpose(2, 1)
        ).inverse()

    def get_all_edge_scalings(self, frame_edge_list):
        scale = torch.cat(
            [self.frames[idx[0]].edges[idx[1]].scaling[None] for idx in frame_edge_list]
        )

        trafo_mat = (
            torch.eye(4, device=self.device)[None]
            * torch.cat(
                [scale, torch.full([len(scale), 1], 1.0, device=self.device)], dim=1
            )[:, None]
        )
        return Transform3d(matrix=trafo_mat)

    def init_blank_frames(
        self,
        image_ls: List = None,
        camera_node: NeuralCamera = None,
        far: float = 150.0,
        near: float = 0.5,
        box_scale: float = 1.5,
        scene_id: int = None,
        cam_pose=None,
    ):
        roty = lambda a: np.array(
            [
                [np.cos(a), 0.0, np.sin(a), 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-np.sin(a), 0.0, np.cos(a), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        if scene_id is None:
            if len(self.scene_frames) > 0:
                scene_id = max(list(self.scene_frames.keys())) + 1
            else:
                scene_id = 0

        root_id = list(self.nodes["background"].keys())[0]
        background_node = list(self.nodes["background"].values())[0]

        if camera_node is None:
            # Takes the first camera node in the scene
            cam_id = list(self.nodes["camera"].keys())[0]
            cam = self.nodes["camera"][cam_id]

        else:
            cam_id = camera_node.scene_idx
            cam = camera_node

        # Place camera in the center of the world frame pointing in positive y axis
        if cam_pose is None:
            cam_pose = np.array(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        # cam_pose = np.matmul(roty(np.deg2rad(180)), cam_pose)

        # Initilize new scene from anchors and known nodes
        scene_dict = {
            "box_scale": box_scale,
            "far_plane": far,
            "first_frame": 0,
            "last_frame": len(image_ls) - 1,
            "near_plane": near,
            "new_world": True,
            "scene_id": scene_id,
            "type": "waymo",
        }

        scene_graph_ls = []
        # add camera poses
        scene_graph_ls.append([root_id, cam_id, cam_pose])

        frame_list = []

        for img in image_ls:
            frame = Frame(
                root_node=background_node,
                image=[img],
                input_graph=scene_graph_ls,
                camera=self.nodes["camera"],
                nodes=self.nodes,
            )
            frame_list.append(frame)

        old_frame_idx = set(self.frames.keys())

        self.scene_frames[scene_id] = []
        self.updateFrames(frame_list, scene_id=scene_id, scene_dict=scene_dict)
        self.scene_frames[scene_id] = np.array(self.scene_frames[scene_id])
        self.frame_ids = list(self.frames.keys())

        new_frames_idx = old_frame_idx.symmetric_difference(set(self.frames.keys()))
        return list(new_frames_idx)

    def add_new_obj_edges(
        self,
        frame_id: int = None,
        object_node: SceneObject = None,
        translation=torch.tensor([0.0, 0.0, 0.0]),
        rotation=torch.tensor([0.0, 0.0, 0.0]),
        box_size_scaling: float = 1.5,
    ):

        frame = self.frames[frame_id]
        root_id = frame.background_node
        obj_class_name = object_node.object_class_name
        scene_object_id = object_node.scene_idx

        obj_class_id = None
        for i, v in self.nodes["object_class"].items():
            if v.name == obj_class_name:
                obj_class_id = i

        if obj_class_id is None:
            # TODO: Debug
            ValueError("A class for {} does not exist.".format(obj_class_name))
            node = [self.createObjectClass(name=obj_class_name)]
            self.updateNodes(
                [{"type_idx": i, "node_type": "class_object", "node": node}]
            )
        # clone().detach().requires_grad_(True)
        rot_mat = so3_exponential_map(rotation[None])[0]
        translation = translation.to(device=self.device)
        object_transformation = torch.cat([rot_mat, translation[:, None]], dim=1)

        object_scaling = (
            torch.eye(3, device=self.device)
            * 1
            / ((object_node.box_size * box_size_scaling) / 2)
        )

        # world to object
        w2o = [root_id, scene_object_id, object_transformation]
        # object to class
        o2c = [scene_object_id, obj_class_id, object_scaling]

        edges = frame.graphFromList([w2o, o2c])
        frame.add_edges(edges)

    def debug_plot_dataset(self, dataset):
        plt_objs = True
        waymo_pose_debug = False
        ADJUST_ARROW_SIZES = True
        ARROW_SIZE_FACTOR = 18

        for ax_id_num,ax_id in enumerate([[0, 1], [0, 2]]):
            plt.figure()
            translation = dataset.poses[:, :3, 3]

            rotation = dataset.poses[:, :3, :3]
            if dataset.type != 'kitti' and waymo_pose_debug:
                rotation = rotation

            # plt.scatter(translation[:, ax_id[0]], translation[:, ax_id[1]], c='cyan')
            plt.axis('equal')
            num_cameras = 5
            iter = [45] #(1,10),(1,7)
            # iter = [89] #(1,17)
            iter = [0, 2, 4, 6, 8]
            # iter += [i - 10 for i in iter]
            for k in iter:
                # co_frame_cam_iters = [i for i in range(k//num_cameras*num_cameras,(k//num_cameras+1)*num_cameras)]
                co_frame_cam_iters = [k%(translation.shape[0]//num_cameras)+i*translation.shape[0]//num_cameras for i in range(num_cameras)]
                # adjacent_frame_cam_iters = [i for i in range(co_frame_cam_iters[0]-num_cameras,co_frame_cam_iters[0])]+[i for i in range(co_frame_cam_iters[-1]+1,co_frame_cam_iters[-1]+1+num_cameras)]
                adjacent_frame_cam_iters = [v-1 for v in co_frame_cam_iters]+[v+1 for v in co_frame_cam_iters]
                adjacent_frame_cam_iters = [v for v in adjacent_frame_cam_iters if 0<=v<translation.shape[0]]
                adjacent_frame_cam_iters = []
                for idx in co_frame_cam_iters+adjacent_frame_cam_iters:
                    plt.scatter(translation[idx, ax_id[0]], translation[idx, ax_id[1]], c='red' if k==idx else 'cyan')
                    for i in range(3):
                        ax = rotation[:, i] * 0.1
                        c = [0, 0, 0]
                        c[i] = 1
                        if ADJUST_ARROW_SIZES:
                            plt.arrow(translation[idx, ax_id[0]], translation[idx, ax_id[1]], ARROW_SIZE_FACTOR*ax[idx, ax_id[0]], ARROW_SIZE_FACTOR*ax[idx, ax_id[1]],
                                color=c)
                        else:
                            plt.arrow(translation[idx, ax_id[0]], translation[idx, ax_id[1]], ax[idx, ax_id[0]], ax[idx, ax_id[1]],
                                color=c)
                if plt_objs:
                    # plt_obj_poses = dataset.visible_objects[k][:, 7:10]
                    plt_obj_poses = dataset.obj_poses[k][:, :3]
                    # plt_obj_yaw = dataset.visible_objects[k][:, 10]
                    plt_obj_yaw = dataset.obj_poses[k][:, 3]
                    plt_mask = np.where(dataset.visible_objects[k][:, -1] != -1)
                    if len(plt_mask[0]) > 0:
                        plt_obj_poses = plt_obj_poses[plt_mask]
                        plt_obj_yaw = plt_obj_yaw[plt_mask]
                        plt.scatter(plt_obj_poses[:, ax_id[0]], plt_obj_poses[:, ax_id[1]], c='orange')

                        if ax_id[1] == 2:
                            plt_obj_rotation = np.array([[np.cos(plt_obj_yaw), -np.sin(plt_obj_yaw)],
                                                            [np.sin(plt_obj_yaw), np.cos(plt_obj_yaw)]])
                            plt_obj_rotation = np.moveaxis(plt_obj_rotation, -1, 0)
                            for o in range(len(plt_obj_rotation)):
                                plt.arrow(plt_obj_poses[o, ax_id[0]], plt_obj_poses[o, ax_id[1]],
                                            plt_obj_rotation[o, 0, 0], plt_obj_rotation[o, 0, 1])
            assert len(iter)==1
            plt.title('(obj from img #%d)'%(k))
            plt.savefig('%s.png'%('BEV' if ax_id==[0,2] else 'side'))
        np.matmul(np.linalg.inv(rotation[0]), rotation)
        dataset.poses[:, :3, :3] = rotation

        return dataset


# Methods to create predefined scenes nodes and take care of all necessary properties
def createNewWorld(global_transformation, scene_no, near_plane, far_plane, lightfield_config):
    # TODO: Remove lagacy background stuff such as near and far plane
    background_node = Background(global_transformation, near_plane, far_plane,
                                 lightfield_config=lightfield_config)
    node_dict = addNode(background_node, "background", type_idx=scene_no)
    return [node_dict]


def createCamera(H, W, focal, type=None, type_idx=None):
    camera_node = NeuralCamera(H, W, focal, type=type)
    node_dict = addNode(camera_node, "camera", type_idx=type_idx)

    return [node_dict]


def createAllCameras(H, W, focal, type_idx=None, type=None):
    nodes_list = []
    for cam_key in focal.keys():
        camera_node = NeuralCamera(H[cam_key], W[cam_key], focal[cam_key], name=cam_key, type=type)
        nodes_list.append(addNode(camera_node,'camera',type_idx=type_idx))

    return nodes_list


def createAllLidars(dataset=None, type_idx=None):
    if dataset.type == 'kitti':
        Tr_li2cam = None
        if hasattr(dataset, 'calibration'):
            calibration = dataset.calibration
            if 'Tr_velo2cam' in calibration:
                Tr_li2cam = calibration['Tr_velo2cam']

        lidar_node = Lidar(Tr_li2cam=Tr_li2cam, name='TOP')
        node_dict = addNode(lidar_node, 'lidar', type_idx=type_idx)
        node_list = [node_dict]
    else:
        node_list = []
        for name in dataset.laser_name:
            lidar_node = Lidar(Tr_li2cam=np.eye(4), name=name)
            node_list.append(addNode(lidar_node, 'lidar', type_idx=type_idx))

    return node_list


def createStereoCamera(H, W, focal, dataset=None, type_idx=None):
    P0 = None
    P1 = None
    if hasattr(dataset, 'calibration'):
        calibration = dataset.calibration
        if 'P2' in calibration:
            P0 = calibration['P2']
            P1 = calibration['P3']
            R0_rect = calibration['Tr_cam2camrect']
            P0 = np.matmul(P0, R0_rect)
            P1 = np.matmul(P1, R0_rect)

    camera_node_0 = NeuralCamera(H, W, focal, P=P0)
    node_dict_1 = addNode(camera_node_0, "camera", type_idx=type_idx)

    camera_node_1 = NeuralCamera(H, W, focal, intrinsics=camera_node_0.intrinsics, P=P1)
    node_dict_2 = addNode(camera_node_1, "camera", type_idx=type_idx)

    return [node_dict_1, node_dict_2]


def createObjectClass(name, type_idx=None):
    # TODO: Add all relevant params to scene init
    object_class_node = ObjectClass(name)
    node_dict = addNode(object_class_node, "object_class", type_idx=type_idx)

    return node_dict


def createSceneObject(length, height, width, object_class_node, type_idx=None):
    scene_object_node = SceneObject(length, height, width, object_class_node)
    node_dict = addNode(scene_object_node, node_type="scene_object", type_idx=type_idx)
    return [node_dict]


# Methods to create, read and write nodes to the scenes
def addNode(
    node,
    node_type=None,
    type_idx=None,
    store_by_name=False,
):
    """
    Adds a node to the Scene's node dictionaries

    Args:
        node: Node object of any node type
        node_type: [str], Type of Node (background, camera, scene_object, object_class)
        type_idx: [int], Integer index unique to the type of nodes
        store_by_name: [bool], If true, nodes are added to the dict with their name instead of type_idx
    """
    node.node_type = node_type
    node.type_idx = type_idx

    # Check if node has type already specified
    if node_type is None:
        if hasattr(node, "node_type"):
            node_type = node.node_type
        else:
            error_msg = "node.node_type has to be set for 'node_type' = ", node_type
            raise TypeError(error_msg)

    elif node_type != node.node_type:
        print(
            "node_type = %s and node.node_type = %s do not match. "
            "Not changing node.node_type, but saving node in %s."
            % (node_type, node.node_type, node_type)
        )

    return {"type_idx": type_idx, "node_type": node_type, "node": node}


def nodes_from_dataset(dataset, neural_scene, object_class_list=None, lightfield_config={}):
    node_list = []
    # Add background
    node_list += createNewWorld(
        dataset.poses[0],
        dataset.scene_id,
        near_plane=dataset.near_plane,
        far_plane=dataset.far_plane,
        lightfield_config=lightfield_config
    )
    # Add Cameras
    node_list += createAllCameras(neural_scene.H, neural_scene.W, neural_scene.focal, type=dataset.type)

    # Add Generic Lidar Node
    node_list += createAllLidars(dataset=dataset)

    # Add object classes
    if hasattr(dataset, 'scene_classes'):
        class2node = {}
        for object_class in dataset.scene_classes:

            # ignore those not in object_class_list
            if object_class_list is not None and int(object_class) not in object_class_list:
                continue
            if object_class == -1:
                continue
            name = _obj_class_label2sem[int(object_class)]
            class_node = createObjectClass(name, type_idx=int(object_class))
            class2node[int(object_class)] = class_node["node"]

            node_list += [class_node]

    # Add objects
    if hasattr(dataset, 'objects_meta'):
        # TODO: Add check if object class already exists like in prev versions
        for object_key, object_val in dataset.objects_meta.items():
            length, height, width, object_class = (
                object_val[1],
                object_val[2],
                object_val[3],
                int(object_val[4]),
            )
            if object_class == -1:
                continue
            # ignore those not in object_class_list
            if object_class_list is not None and int(object_class) not in object_class_list:
                continue
            node_list += createSceneObject(
                length,
                height,
                width,
                class2node[int(object_class)],
                type_idx=int(object_key),
            )

    return node_list


def frames_from_dataset(dataset, nodes, n_frames=None, veh_world=False, all_cams=False, type='waymo'):
    # TODO: Rewrite when new dataset class is implemented
    n_img = dataset.num_cam_frames
    n_pcd = len(dataset.point_cloud_pth)

    cam_ids = [c.scene_idx for c in nodes['camera'].values()]
    n_cameras = dataset.num_cameras

    laser_ids = [c.scene_idx for c in nodes['lidar'].values()]
    n_lasers = dataset.num_lasers

    # [n_img, 4, 4] --> [n_cameras, n_frames, 16]
    camera_poses = dataset.poses.reshape([n_cameras,n_img//n_cameras, -1])

    # [n_img] --> [n_cameras, n_frames]
    images = [dataset.images[n_img//n_cameras*i:n_img//n_cameras*(i+1)] for i in range(n_cameras)]

    # [n_pcd] --> [n_lasers, n_frames]
    pcd_path = [dataset.point_cloud_pth[n_pcd // n_lasers * i:n_pcd // n_lasers * (i + 1)] for i in range(n_lasers)]

    # Extract objects from dataset
    # object_poses = dataset.obj_poses

    if veh_world:
        # TODO: Rewrite together with loader such that lidar_poses_world and camera_poses_world are generated here and not inside the dataloader
        lidar_poses = dataset.lidar_poses_world.reshape([n_lasers, n_pcd // n_lasers, -1])
        camera_poses = dataset.poses_world.reshape([n_cameras, n_img // n_cameras, -1])
        if type == 'waymo':
            if not all_cams:
                for pop_cam in cam_ids[-4:]:
                    nodes["camera"].pop(pop_cam)
            else:
                for pop_cam in cam_ids[-2:]:
                    nodes["camera"].pop(pop_cam)
        if type == 'algolux':
            if not all_cams:
                for pop_cam in cam_ids[-1:]:
                    nodes["camera"].pop(pop_cam)
    else:
        lidar_poses = dataset.lidar_poses.reshape([n_lasers, n_pcd // n_lasers, -1])

    if hasattr(dataset, 'obj_poses'):
        EXAMINE_ALL_CAM_OBJECTS = (dataset.type != 'kitti')
        if EXAMINE_ALL_CAM_OBJECTS:
            object_poses = []
            for frame_id in range(n_img // n_cameras):
                cand_poses = np.concatenate([dataset.obj_poses[frame_id+i*n_img//n_cameras] for i in range(n_cameras)],0)
                cand_poses = cand_poses[np.where(cand_poses[:, -1] != -1)]
                obj_pose,added_IDs = [],set()
                for pos in cand_poses:
                    if pos[4] >= 0 and pos[4] not in added_IDs:
                        obj_pose.append(pos)
                        added_IDs.add(pos[4])
                if len(obj_pose) == 0:
                    obj_pose =[np.ones(6, ) * -1]
                object_poses.append(np.stack(obj_pose,0))
        else:
            object_poses = dataset.obj_poses[:n_img//n_cameras] #In the Kitti case, assuming the same objects appear in both cameras, so only taking objects from the entries corresponding to the first camera.

    # Get relevant Ids
    assert len(nodes["background"]) == 1
    background_node = list(nodes["background"].values())[0]
    root_id = background_node.scene_idx
    cam_ids = [c.scene_idx for c in nodes["camera"].values()]
    n_cameras = len(cam_ids)
    n_img = n_cameras * len(images[0])
    if lidar_poses is not None:
        lidar_ids = [li.scene_idx for li in nodes['lidar'].values()]

    frame_list = []
    # TODO: Get real frame number of the dataset after refacrtoring the dataset class
    for frame_id in tqdm.tqdm(
        range(n_img // n_cameras), desc="Frames from Dataset", leave=False
    ):
        image_ls = [images[i][frame_id] for i in range(n_cameras)]
        point_cloud_pth = [pcd_path[i][frame_id] for i in range(n_lasers)]

        scene_graph_ls = []
        # add camera poses
        for c in range(n_cameras):
            scene_graph_ls.append([root_id, cam_ids[c], camera_poses[c, frame_id]])

        # add lidar if available
        if lidar_poses is not None:
            # scene_graph_ls.append([root_id, lidar_id, lidar_poses[frame_id]])
            for li in range(n_lasers):
                scene_graph_ls.append([root_id, lidar_ids[li], lidar_poses[li, frame_id]])

        if hasattr(dataset, 'obj_poses'):
            type2object = {n.type_idx: n for n in nodes["scene_object"].values()}
            type2class = {n.type_idx: n for n in nodes["object_class"].values()}
            has_objects = False

            for obj_pose in object_poses[frame_id]:
                type_idx = int(obj_pose[5])
                if type_idx >= 0:
                    if type_idx not in type2object:
                        # print(f'Warning: skipping {type_idx}')
                        continue
                    has_objects = True
                    scene_object = type2object[type_idx]
                    scene_idx = scene_object.scene_idx
                    object_class_type_idx = scene_object.object_class_type_idx
                    obj_size = scene_object.box_size

                    obj_class_scene_id = type2class[object_class_type_idx].scene_idx

                    ######################
                    world2object = obj_pose[:4]
                    # TODO: Test and fix in dataset when finished
                    yaw = -torch.tensor(world2object[-1])
                    if yaw < 1e-4:
                        yaw += 1e-4
                    # rot = euler_angles_to_matrix(torch.tensor([0., yaw, 0.]), 'ZYX')
                    t = torch.tensor(world2object[:-1])
                    obj_rot = euler_angles_to_matrix(torch.tensor([0., 0., yaw]), 'XYZ')
                    obj_trafo = torch.cat([obj_rot, t[:, None]], dim=1)
                    obj_trafo = torch.cat([obj_trafo, torch.tensor([0., 0., 0., 1.])[None]])
                    object_transformation = obj_trafo.flatten().to(torch.float32)

                    ######################
                    object_scaling = (
                        torch.eye(3, device="cpu") * 1 / (obj_size * dataset.box_scale / 2)
                    )

                    # world to object
                    w2o = [root_id, scene_idx, object_transformation]
                    # object to camera
                    o2c = [scene_idx, obj_class_scene_id, object_scaling]
                    scene_graph_ls.append(w2o)
                    scene_graph_ls.append(o2c)
        else:
            has_objects = False

        if has_objects is False and hasattr(dataset, 'obj_poses'):
            continue
        frame = Frame(
            root_node=background_node,
            image=image_ls,
            input_graph=scene_graph_ls,
            camera=nodes["camera"],
            lidar=nodes.get("lidar", None),
            nodes=nodes,
            point_cloud_pth=point_cloud_pth,
            global_transformation=dataset.veh_pose[frame_id] if dataset.type == 'waymo' or dataset.type == 'algolux' else None,
        )

        frame_list += [frame]

        if n_frames is not None and (frame_id + 1) >= n_frames:
            break

    return frame_list


def get_obj_inputs(
    scene,
    local_pts_bundle,
    xycfn,
    use_locations=False,
    _transient_head=False,
):
    local_pts_idx, local_pts, local_dirs = local_pts_bundle
    xycfn_relevant = xycfn[local_pts_idx]
    # Get informations from all object nodes
    class_idx, object_idx = get_object_node_information(
        scene, obj_ids=xycfn_relevant.unique()
    )

    # Car object class
    node_idx = class_idx[0]
    assert len(class_idx.unique()) == 1

    # Get all object nodes that are connected to the object class at node_idx
    class2obj_idx = torch.where(class_idx == node_idx)

    # Get the index to all points that intersect with a node of this class by checking for the node in all intersection points
    intersections_obj_idx = [
        torch.where(xycfn_relevant == i)[0] for i in object_idx[class2obj_idx]
    ]
    model_intersection_idx = torch.cat(intersections_obj_idx)

    # Get the idx of rays intersecting this model node in the ray bundle
    local_pts_model_idx = tuple(
        idx[model_intersection_idx] for idx in local_pts_idx[:-1]
    )

    # Get the intersection points and ray directions in the local coordinate frames
    p = local_pts.reshape(-1, 3)[model_intersection_idx]
    d = local_dirs.reshape(-1, 3)[model_intersection_idx]

    if use_locations:
        locations_obj = node_locations[model_intersection_idx]
    else:
        locations_obj = None

    if _transient_head:
        image_latent_obj = node_image_latents[model_intersection_idx]
    else:
        image_latent_obj = None

    # Add the respective latent codes to the models input
    # input_latent = torch.cat(
    #     [
    #         object_latent[class2obj_idx][k][None].repeat(len(pts_idx), 1)
    #         for k, pts_idx in enumerate(intersections_obj_idx)
    #     ]
    # )

    # get object latents
    object_latent = torch.stack(
        [scene.nodes[int(n)]["node"].latent for n in object_idx]
    )
    # Construct Input
    # inputs = torch.cat(
    #     [x, input_latent, d, locations_obj, image_latent_obj],
    #     dim=-1,
    # )

    return {
        "pts": p,
        "dirs": d,
        "locs": locations_obj,
        "object_latent": object_latent,
        "image_latent": image_latent_obj,
        "local_pts_model_idx": local_pts_model_idx,
        "node_class_idx": node_idx,
        "object_idx": object_idx,
        "intersections_obj_idx": intersections_obj_idx,
    }


def get_object_node_information(scene, obj_ids):
    """

    Args:
        scene:
        obj_ids:
        relevant_xycfn:
    Returns:
        class_idx:
        object_idx:
        object_latent:
    """
    # Get all latent codes and the index to the respective models
    class_idx = []
    object_idx = []
    # object_latent = []
    # Run time n_class
    for n in obj_ids:
        obj_node = scene.nodes[int(n)]["node"]
        assert scene.nodes[int(n)]["node_type"] == "scene_object"
        class_node = scene.type2class[obj_node.object_class_type_idx]

        class_idx.append(class_node.scene_idx)
        object_idx.append(n)
        # object_latent.append(obj_node.latent)

    class_idx = torch.tensor(class_idx)
    object_idx = torch.tensor(object_idx)
    # object_latent = torch.stack(object_latent)

    return class_idx, object_idx
