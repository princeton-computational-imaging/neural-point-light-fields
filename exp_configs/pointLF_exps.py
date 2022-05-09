from haven import haven_utils as hu

EXP_GROUPS = {}


def get_scenes(scene_ids, selected_frames='default', dataset='waymo', object_types=None):
    scene_list = []
    for scene_id in scene_ids:
        first_frame = None
        last_frame = None
        if isinstance(selected_frames, list):
            first_frame = selected_frames[0]
            last_frame = selected_frames[1]

        if type(scene_id) != list and dataset == 'waymo':
            for record_id in range(25):
                scene_list += [{'scene_id': [scene_id, record_id],
                                'type': dataset,
                                'first_frame': None,
                                'last_frame': None,
                                'far_plane': 150,
                                'near_plane': .5,
                                "new_world": True,
                                'box_scale': 1.5,
                                'object_types': object_types,
                                'fix': True,
                                'pt_cloud_fix': True}]

        else:
            scene_dict = {'scene_id': scene_id,
                          'type': dataset,
                          'first_frame': first_frame,
                          'last_frame': last_frame,
                          'far_plane': 150,
                          'near_plane': .5,
                          "new_world": True,
                          'box_scale': 1.5,
                          'object_types': object_types,
                          'fix': True,
                          'pt_cloud_fix': True}
            if object_types is not None:
                scene_dict['object_types'] = object_types
            scene_list += [scene_dict]
    return scene_list


EXP_GROUPS['pointLF_waymo_local'] = hu.cartesian_exp_group({
    "n_rays": [512],
    'image_batch_size': [2, ],
    "chunk": [512],
    "scenes": [
        # get_scenes(scene_ids=[[0, 8]], dataset='waymo', selected_frames=[135, 197], object_types=['TYPE_VEHICLE']),
        # get_scenes(scene_ids=[[0, 11]], dataset='waymo', selected_frames=[0, 20], object_types=['TYPE_VEHICLE']),
        get_scenes(scene_ids=[[0, 2]], dataset='waymo', selected_frames=[0, 80], object_types=['TYPE_VEHICLE']),
        # get_scenes(scene_ids=[[0, 2]], dataset='waymo', selected_frames=[0, 197], object_types=['TYPE_VEHICLE']),
        # get_scenes(scene_ids=[[2, 0]], dataset='waymo', selected_frames=[0, 10], object_types=['TYPE_VEHICLE']),
        # get_scenes(scene_ids=[[0, 19]], dataset='waymo', selected_frames=[0, 198], object_types=['TYPE_VEHICLE']),
        # get_scenes(scene_ids=[[0, 19]], dataset='waymo', selected_frames=[96, 198], object_types=['TYPE_VEHICLE']),
        # get_scenes(scene_ids=[[1, 15]], dataset='waymo', selected_frames=[0, 197], object_types=['TYPE_VEHICLE']),
        # get_scenes(scene_ids=[[2, 0]], dataset='waymo', selected_frames=[0, 40], object_types=['TYPE_VEHICLE']),
    ],
    "precache": [False],
    "lrate": [0.001, ],
    "lrate_decay": 250,
    "netchunk": 65536,
    'lightfield': {'k_closest': 4,
                   'n_features': 128,
                   # 'n_sample_pts': 20000,
                   'n_sample_pts': 5000,
                   # 'pointfeat_encoder': 'pointnet_lf_global_weighted',
                   # 'pointfeat_encoder': 'multiview_attention_modulation',
                   'pointfeat_encoder': 'multiview_attention',
                   # 'pointfeat_encoder': 'multiview_attention_up',
                   # 'pointfeat_encoder': 'naive_ablation',
                   # 'pointfeat_encoder': 'one_point_ablation',
                   # 'pointfeat_encoder': 'pointnet_ablation',
                   # 'pointfeat_encoder': 'encoding_attention_only',
                   # 'pointfeat_encoder': 'multiview_attention_big',
                   # 'pointfeat_encoder': 'multiview_encoded',
                   # 'pointfeat_encoder': 'multiview_distance_attention',
                   # 'pointfeat_encoder': 'multiview_encoded_modulation',
                   # 'pointfeat_encoder': 'multiview_encoded_weighted_modulation',
                   'merge_pcd': False,
                   'all_cams': True,
                   'D_lf': 8,
                   'skips_lf': [4],
                   'camera_centered': False,
                   'augment_frame_order': True,
                   'new_enc': False,
                   'torch_sampler': True,
                   'sky_dome': True,
                   'num_merged_frames': 1,
                   },
    # "overfit": "frame",
    "scale": 0.0625, # 0.125,
    "point_chunk": 1e7,
    'version': [0],
    'tonemapping': False,
    'pose_refinement': False,
    'pt_cache': True,
},
    remove_none=True
)


EXP_GROUPS['pointLF_waymo_server'] = hu.cartesian_exp_group({
    "n_rays": [8192],
    'image_batch_size': [2, ],
    "chunk": [64000],
    "scenes": [
        # get_scenes(scene_ids=[[0, 2]], dataset='waymo', selected_frames=[0, 20], object_types=['TYPE_VEHICLE']),
        # get_scenes(scene_ids=[[0, 2]], dataset='waymo', selected_frames=[0, 40], object_types=['TYPE_VEHICLE']),
        get_scenes(scene_ids=[[0, 2]], dataset='waymo', selected_frames=[0, 80], object_types=['TYPE_VEHICLE']),
        # get_scenes(scene_ids=[[0, 2]], dataset='waymo', selected_frames=[0, 120], object_types=['TYPE_VEHICLE']),
        # get_scenes(scene_ids=[[0, 2]], dataset='waymo', selected_frames=[0, 197], object_types=['TYPE_VEHICLE']),
        # get_scenes(scene_ids=[[0, 8]], dataset='waymo', selected_frames=[171, 190], object_types=['TYPE_VEHICLE']),
        # get_scenes(scene_ids=[[0, 8]], dataset='waymo', selected_frames=[135, 197], object_types=['TYPE_VEHICLE']),
        # get_scenes(scene_ids=[[0, 11]], dataset='waymo', selected_frames=[0, 164], object_types=['TYPE_VEHICLE']),
        # get_scenes(scene_ids=[[1, 15]], dataset='waymo', selected_frames=[0, 197], object_types=['TYPE_VEHICLE']),
        # get_scenes(scene_ids=[[2, 3]], dataset='waymo', selected_frames=[0, 198], object_types=['TYPE_VEHICLE']),
        # get_scenes(scene_ids=[[2, 9]], dataset='waymo', selected_frames=[0, 197], object_types=['TYPE_VEHICLE']),
    ],
    "precache": [False],
    "latent_balance": 0.0001,
    "lrate_decay": 250,
    "netchunk": 65536,
    'lightfield': {'k_closest': 8,
                   'n_features': 128,
                   'n_sample_pts': 500,
                   'pointfeat_encoder': 'multiview_attention',
                   'merge_pcd': True,
                   'all_cams': False,
                   'D_lf': 8,
                   'W_lf': 256,
                   'skips_lf': [4],
                   'layer_modulation': False,
                   'camera_centered': False,
                   'augment_frame_order': True,
                   'new_enc': False,
                   'torch_sampler': True,
                   'sky_dome': True,
                   'num_merged_frames': 20,
                   },
    # "overfit": "frame",
    "scale": .125,
    'version': 0,
    "point_chunk": 1e7,
    'tonemapping': False,
    'pose_refinement': False,
    'pt_cache': True,
},
    remove_none=True
)
