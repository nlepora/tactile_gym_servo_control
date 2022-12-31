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

from tactile_gym_servo_control.utils_robot_sim.setup_embodiment_env import setup_embodiment_env
from tactile_gym_servo_control.collect_data.setup_collect_sim_data import setup_collect_data

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def collect_data(
    embodiment,
    target_df,
    image_dir
):

    hover = [0, 0, 7.5, 0, 0, 0]

    # start 30mm above workframe origin
    embodiment.move_linear([0, 0, 30, 0, 0, 0])

    # ==== data collection loop ====
    for _, row in target_df.iterrows():
        i_obj = int(row.loc["obj_id"])
        i_pose = int(row.loc["pose_id"])
        pose = row.loc["pose_1":"pose_6"].values
        move = row.loc["move_1":"move_6"].values
        obj_pose = row.loc["obj_pose"]
        sensor_image = row.loc["sensor_image"]

        with np.printoptions(precision=2, suppress=True):
            print(f"Collecting data for object {i_obj}, pose {i_pose}: ...")

        # pose is relative to object
        pose += obj_pose

        # move to above new pose (avoid changing pose in contact with object)
        embodiment.move_linear(pose - move + hover)
 
        # move down to offset position
        embodiment.move_linear(pose - move)

        # move to target positon inducing shear effects
        embodiment.move_linear(pose)

        # process tactile image
        image_outfile = os.path.join(image_dir, sensor_image)
        embodiment.sensor_process(outfile=image_outfile)

        # move to target positon inducing shear effects
        embodiment.move_linear(pose + hover)

    # finish 30mm above workframe origin
    embodiment.move_linear([0, 0, 30, 0, 0, 0])
    embodiment.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--tasks',
        nargs='+',
        help="Choose task from [surface_3d edge_2d edge_3d edge_5d].",
        default=['edge_2d']
    )

    # parse arguments
    args = parser.parse_args()
    tasks = args.tasks
    
    for task in tasks:

        target_df, image_dir, env_params, tactip_params = \
            setup_collect_data[task]()

        embodiment = setup_embodiment_env(
            **env_params, 
            tactip_params = tactip_params, #quick_mode=True 
        )

        collect_data(
            embodiment,
            target_df,
            image_dir
        )
