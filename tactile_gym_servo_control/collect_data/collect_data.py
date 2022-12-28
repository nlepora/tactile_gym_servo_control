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

from tactile_gym_servo_control.utils.load_embodiment_and_env import POSE_UNITS
from tactile_gym_servo_control.utils.load_embodiment_and_env import load_embodiment_and_env
from tactile_gym_servo_control.collect_data.setup_collect_data import setup_collect_data

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def collect_data(
    embodiment,
    target_df,
    image_dir
):

    hover = [0, 0, 7.5, 0, 0, 0] * POSE_UNITS

    # move to workframe origin
    embodiment.move([0, 0, 0, 0, 0, 0])

    # ==== data collection loop ====
    for _, row in target_df.iterrows():
        i_obj = int(row.loc["obj_id"])
        i_pose = int(row.loc["pose_id"])
        pose = row.loc["pose_1":"pose_6"].values.astype(np.float32)
        move = row.loc["move_1":"move_6"].values.astype(np.float32)
        obj_pose = row.loc["obj_pose"]
        sensor_image = row.loc["sensor_image"]

        with np.printoptions(precision=2, suppress=True):
            print(f"Collecting data for object {i_obj}, pose {i_pose}: ...")
            
        # convert to pybullet units
        pose *= POSE_UNITS
        move *= POSE_UNITS

        # pose is relative to object
        pose += obj_pose

        # move to above new pose (avoid changing pose in contact with object)
        embodiment.move(pose - move - hover)
 
        # move down to offset position
        embodiment.move(pose - move)

        # move to target positon inducing shear effects
        embodiment.move(pose)

        # process frames
        img = embodiment.process_sensor()

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
        default=['surface_3d']
    )

    # parse arguments
    args = parser.parse_args()
    tasks = args.tasks
    
    for task in tasks:

        target_df, image_dir, env_params = setup_collect_data[task]()

        embodiment = load_embodiment_and_env(
            **env_params
        )

        collect_data(
            embodiment,
            target_df,
            image_dir
        )
