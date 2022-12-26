import os
import numpy as np
import pandas as pd
import json

from tactile_gym_servo_control.utils.pybullet_utils import setup_pybullet_env
from tactile_gym.utils.general_utils import check_dir

stimuli_path = os.path.join(os.path.dirname(__file__), '../stimuli')


def load_embodiment_and_env(
    stim_name="square", 
    show_gui=True, 
    show_tactile=True,
    quick_mode=False
):

    assert stim_name in ["square", "foil", "clover", "circle",
                         "saddle"], "Invalid Stimulus"

    tactip_params = {
        "name": "tactip",
        "type": "standard",
        "core": "no_core",
        "dynamics": {},
        "image_size": [128, 128],
        "turn_off_border": False,
    }

    # setup stimulus
    stimulus_pos = [0.6, 0.0, 0.0125]
    stimulus_rpy = [0, 0, 0]
    stim_path = os.path.join(
        stimuli_path,
        stim_name,
        stim_name + ".urdf"
    )

    # set the work frame of the robot (relative to world frame)
    workframe_pos = [0.6, 0.0, 0.0525]
    workframe_rpy = [-np.pi, 0.0, np.pi / 2]

    # setup robot data collection env
    embodiment, _ = setup_pybullet_env(
        stim_path,
        tactip_params,
        stimulus_pos,
        stimulus_rpy,
        workframe_pos,
        workframe_rpy,
        show_gui,
        show_tactile,
    )

    def move_linear(pose):
        pos = pose[:3]
        rpy = pose[3:]
        embodiment.move_linear(pos, rpy, quick_mode=quick_mode)

    embodiment.move = move_linear

    return embodiment


def make_target_df_rand(
    num_poses, 
    shuffle_data=False,
    pose_llims=[0, 0, 0, 0, 0, 0], 
    pose_ulims=[0, 0, 0, 0, 0, 0], 
    move_llims=[0, 0, 0, 0, 0, 0], 
    move_ulims=[0, 0, 0, 0, 0, 0], 
    obj_poses=[[0, 0, 0, 0, 0, 0]]  
):

    # generate random poses
    np.random.seed()
    poses = np.random.uniform(
        low=pose_llims, high=pose_ulims, size=(num_poses, 6)
    )
    poses = poses[np.lexsort((poses[:, 1], poses[:, 5]))]
    moves = np.random.uniform(
        low=move_llims, high=move_ulims, size=(num_poses, 6)
    )

    # generate and save target data
    target_df = pd.DataFrame(
        columns=[
            "sensor_image",
            "obj_id",
            "obj_pose",
            "pose_id",
            "pose_1", "pose_2", "pose_3", "pose_4", "pose_5", "pose_6",
            "move_1", "move_2", "move_3", "move_4", "move_5", "move_6",
        ]
    )

    # populate dateframe
    for i in range(num_poses * len(obj_poses)):
        image_file = "image_{:d}.png".format(i + 1)
        i_pose, i_obj = (int(i % num_poses), int(i / num_poses))
        pose = poses[i_pose, :]
        move = moves[i_pose, :]
        target_df.loc[i] = np.hstack(
            ((image_file, i_obj + 1, obj_poses[i_obj], i_pose + 1), pose, move)
        )

    if shuffle_data:
        target_df = target_df.sample(frac=1).reset_index(drop=True)

    return target_df


def create_data_dir(
    collect_dir,
    target_df
):

    # experiment metadata
    image_dir = os.path.join(collect_dir, "images")
    target_file = os.path.join(collect_dir, "targets.csv")

    # check save dir exists
    check_dir(collect_dir)

    # create dirs
    os.makedirs(collect_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # save directory names and paths
    paths = locals().copy()
    del paths["target_df"] # not a path
    with open(os.path.join(collect_dir, "paths.json"), 'w') as f:
        json.dump(paths, f)

    # save target csv
    target_df.to_csv(target_file, index=False)

    return image_dir
