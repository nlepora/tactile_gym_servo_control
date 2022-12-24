"""
python collect_data.py -t surface_3d
python collect_data.py -t edge_2d
python collect_data.py -t edge_3d
python collect_data.py -t edge_5d
python collect_data.py -t surface_3d edge_2d edge_3d edge_5d
"""

import os
import argparse
import numpy as np
import cv2

from tactile_gym_servo_control.utils.pybullet_utils import setup_pybullet_env
from tactile_gym_servo_control.data_collection.setup_data_collection import setup_surface_3d_data_collection
from tactile_gym_servo_control.data_collection.setup_data_collection import setup_edge_2d_data_collection
from tactile_gym_servo_control.data_collection.setup_data_collection import setup_edge_3d_data_collection
from tactile_gym_servo_control.data_collection.setup_data_collection import setup_edge_5d_data_collection

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

stimuli_path = os.path.join(os.path.dirname(__file__), "../stimuli")


def load_embodiment_and_env(
    stim_name="square", 
    show_gui=True, 
    show_tactile=True
):

    assert stim_name in ["square", "foil",
                         "clover", "circle",
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

    return embodiment


def collect_data(
    embodiment,
    target_df,
    image_dir,
    quick_mode=False,
):

    hover_dist = 0.0075

    # move to work frame
    embodiment.move_linear([0, 0, 0], [0, 0, 0], quick_mode)

    # ==== data collection loop ====
    for index, row in target_df.iterrows():
        i_obj = int(row.loc["obj_id"])
        i_pose = int(row.loc["pose_id"])
        pose = row.loc["pose_1":"pose_6"].values.astype(np.float32)
        move = row.loc["move_1":"move_6"].values.astype(np.float32)
        obj_pose = row.loc["obj_pose"]
        sensor_image = row.loc["sensor_image"]

        # define the new pos and rpy
        # careful around offset for camera orientation
        obj_pose_array = np.array([float(i) for i in obj_pose])
        pose_array = np.array([float(i) for i in pose])
        move_array = np.array([float(i) for i in move])

        # combine relative pose and object pose
        new_pose = obj_pose_array + pose_array

        # convert to pybullet form
        final_pos = new_pose[:3] * 0.001  # to mm
        final_rpy = new_pose[3:] * np.pi / 180  # to rad
        move_pos = move_array[:3] * 0.001  # to mm
        move_rpy = move_array[3:] * np.pi / 180  # to rad

        with np.printoptions(precision=2, suppress=True):
            print(f"Collecting data for object {i_obj}, pose {i_pose}: ...")

        # move to slightly above new pose (avoid changing pose in contact with object)
        embodiment.move_linear(
            final_pos - move_pos - [0, 0, hover_dist],
            final_rpy - move_rpy,
            quick_mode
        )

        # move down to offset position
        embodiment.move_linear(
            final_pos - move_pos,
            final_rpy - move_rpy,
            quick_mode
        )

        # move to target positon inducing shear effects
        embodiment.move_linear(
            final_pos,
            final_rpy,
            quick_mode
        )

        # process frames
        img = embodiment.process_sensor()

        # raise tip before next move
        embodiment.move_linear(
            final_pos - [0, 0, hover_dist],
            final_rpy,
            quick_mode
        )

        # save image
        image_outfile = os.path.join(image_dir, sensor_image)
        cv2.imwrite(image_outfile, img)

    embodiment.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--tasks',
        nargs='+',
        help="Choose task from ['surface_3d', 'edge_2d', 'edge_3d', 'edge_5d'].",
        default=['edge_3d']
    )

    # parse arguments
    args = parser.parse_args()
    tasks = args.tasks

    setup_data_collection = {
        "surface_3d": setup_surface_3d_data_collection,
        "edge_2d": setup_edge_2d_data_collection,
        "edge_3d": setup_edge_3d_data_collection,
        "edge_5d": setup_edge_5d_data_collection
    }

    num_samples = 10
    collect_dir_name = "example_data"
    
    for task in tasks:

        target_df, image_dir, workframe_pos, workframe_rpy = setup_data_collection[task](
            num_samples=num_samples,
            shuffle_data=False,
            collect_dir_name=collect_dir_name,
        )

        embodiment = load_embodiment_and_env()

        collect_data(
            embodiment,
            target_df,
            image_dir
        )
