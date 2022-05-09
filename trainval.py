import importlib; importlib.util.find_spec("waymo_open_dataset")
# assert importlib.util.find_spec("waymo_open_dataset") is not None, "no waymo"

import argparse
import os
import exp_configs
import pandas as pd
import numpy as np
import torch
import time

from torch.utils.data import DataLoader
from haven import haven_utils as hu
from haven import haven_wizard as hw
from src import models
from src.scenes import NeuralScene
from src import utils as ut
from src import utils_dist as utd

torch.backends.cudnn.benchmark = True

ONLY_PRESENT_SCORES = True

# 1. define the training and validation function
def trainval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """
    # set seed
    seed = 42 + exp_dict.get("runs", 0)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_state_dict = None

    render_epi = False
    if render_epi:
        exp_dict_pth = '/home/julian/workspace/NeuralSceneGraphs/model_library/scene_0_2/exp_dict.json'
        model_pth = '/home/julian/workspace/NeuralSceneGraphs/model_library/scene_0_2/model.pth'
        epi_frame_idx = 6
        epi_row = 96

        # exp_dict_pth = '/home/julian/workspace/NeuralSceneGraphs/model_library/scene_2_3/exp_dict.json'
        # model_pth = '/home/julian/workspace/NeuralSceneGraphs/model_library/scene_2_3/model.pth'
        # epi_frame_idx = 79
        # epi_row = 118
        # epi_row = 101

        if args.render_only and os.path.exists(exp_dict_pth):
            exp_dict = hu.load_json(exp_dict_pth)
            model_state_dict = hu.torch_load(model_pth)

            exp_dict["scale"] = 0.0625
            exp_dict["scale"] = 0.125

    scene = NeuralScene(
        scene_list=exp_dict["scenes"],
        datadir=args.datadir,
        args=args,
        exp_dict=exp_dict,
    )
    
    batch_size = min(len(scene), exp_dict.get("image_batch_size", 1))
    rand_sampler = torch.utils.data.RandomSampler(
        scene, num_samples=args.epoch_size * batch_size, replacement=True
    )

    scene_loader = torch.utils.data.DataLoader(
        scene,
        sampler=rand_sampler,
        collate_fn=ut.collate_fn_dict_of_lists,
        batch_size=batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )

    if scene.refine_camera_pose:
        calib_sampler = torch.utils.data.RandomSampler(
            scene, num_samples=len(scene), replacement=True
        )
        scene_calib_loader = torch.utils.data.DataLoader(
            scene,
            sampler=calib_sampler,
            collate_fn=ut.collate_fn_dict_of_lists,
            batch_size=1,
            num_workers=0,
            drop_last=True,
        )
    # TODO: Find permanent fix https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999
    # torch.multiprocessing.set_sharing_strategy('file_system')

    model = models.Model(scene, exp_dict, precache=exp_dict.get("precache"), args=args)

    # 3. load checkpoint
    chk_dict = hw.get_checkpoint(savedir, return_model_state_dict=True)

    if len(chk_dict["model_state_dict"]):
        model.set_state_dict(chk_dict["model_state_dict"])

    if model_state_dict is not None:
        model.set_state_dict(model_state_dict)

    # val_dict = model.val_on_scene(scene, savedir_images=os.path.join(savedir, "images"), all_frames=True)

    if not args.render_only:
        for e in range(chk_dict["epoch"], 5000):
            # 0. init score dict
            score_dict = {"epoch": e, "n_objects": len(scene.nodes["scene_object"])}

            # (3. Optional Camera Calibration)
            if e % 25 == 0 and scene.refine_camera_pose and e > 0:
                for e_calib in range(5):
                    s_time = time.time()
                    scene.recalibrate = True
                    calib_dict = model.train_on_scene(scene_calib_loader)
                    scene.recalibrate = False
                    score_dict["calib_time"] = time.time() - s_time
                    score_dict.update(calib_dict)

            # 1. train on batch
            s_time = time.time()
            train_dict = model.train_on_scene(scene_loader)
            score_dict["train_time"] = time.time() - s_time
            score_dict.update(train_dict)

            s_time = time.time()
            val_dict = model.val_on_scene(
                scene,
                savedir_images=os.path.join(savedir, "images"),
            )

            # 2. val on batch
            if e % 100 == 0 and e > 0:
                val_dict = model.val_on_scene(
                    scene,
                    savedir_images=os.path.join(savedir, "images_all_frames_{}".format(e)),
                    all_frames=True,
                )

            score_dict["val_time"] = time.time() - s_time
            score_dict.update(val_dict)
            # ONLY MASTER PROCESS?
            if utd.is_main_process():
                # 3. save checkpoint
                chk_dict["score_list"] += [score_dict]
                hw.save_checkpoint(
                    savedir,
                    model_state_dict=model.get_state_dict(),
                    score_list=chk_dict["score_list"],
                    verbose=not ONLY_PRESENT_SCORES,
                )
                if ONLY_PRESENT_SCORES:
                    score_df = pd.DataFrame(chk_dict["score_list"])
                    print("Save directory: %s" % savedir)
                    print(score_df.tail(1).to_string(index=False), "\n")
    elif render_epi:
        val_dict = model.val_on_scene(scene, savedir_images=os.path.join(savedir, "images"), all_frames=False, EPI=True,
                                      epi_row=epi_row, epi_frame_idx=epi_frame_idx)
    else:
        val_dict = model.val_on_scene(scene, savedir_images=os.path.join(savedir, "images"), all_frames=True)


# 7. create main
if __name__ == "__main__":
    # 9. Launch experiments using magic command
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp_group_list", nargs="+", help="Define which exp groups to run."
    )
    parser.add_argument(
        "-sb",
        "--savedir_base",
        default=None,
        help="Define the base directory where the experiments will be saved.",
    )
    parser.add_argument("-d", "--datadir")
    parser.add_argument(
        "-r", "--reset", default=0, type=int, help="Reset or resume the experiment."
    )
    parser.add_argument(
        "-j", "--job_scheduler", default=None, help="Run jobs in cluster."
    )
    parser.add_argument(
        "-v",
        "--visualize",
        default="results/neural_scenes.ipynb",
        help="Run jobs in cluster.",
    )
    parser.add_argument("-p", "--python_binary_path", default="python")
    parser.add_argument("-db", "--debug", type=int, default=0)
    parser.add_argument("--epoch_size", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--render_only", type=bool, default=False)
    # parser.add_argument(
    #     "--dist_url", default="env://", help="url used to set up distributed training"
    # )
    parser.add_argument("--ngpus", type=int, default=1)

    args, others = parser.parse_known_args()

    # Load job config to run things on cluster
    python_binary_path = args.python_binary_path
    jc = None
    if os.path.exists("job_config.py"):
        import job_config

        jc = job_config.JOB_CONFIG
        if args.ngpus > 1:
            jc["resources"]["gpu"] = args.ngpus
            python_binary_path += (
                f" -m torch.distributed.launch --nproc_per_node={args.ngpus} --use_env "
            )

    # utd.init_distributed_mode(args)
    # if args.distributed and not utd.is_main_process():
    #     args.reset = 0

    hw.run_wizard(
        func=trainval,
        exp_groups=exp_configs.EXP_GROUPS,
        savedir_base=args.savedir_base,
        reset=args.reset,
        python_binary_path=python_binary_path,
        job_config=jc,
        args=args,
        use_threads=True,
        results_fname="results/neural_scenes.ipynb",
    )
