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

from tactile_gym_servo_control.data_collection.data_collection_utils import load_embodiment_and_env
from tactile_gym_servo_control.data_collection.setup_data_collection import setup_surface_3d_data_collection
from tactile_gym_servo_control.data_collection.setup_data_collection import setup_edge_2d_data_collection
from tactile_gym_servo_control.data_collection.setup_data_collection import setup_edge_3d_data_collection
from tactile_gym_servo_control.data_collection.setup_data_collection import setup_edge_5d_data_collection

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def collect_data(
    embodiment,
    target_df,
    image_dir
):

    hover = [0, 0, 7.5*1e-3, 0, 0, 0]

    # move to work frame
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
            
        # convert to pybullet form
        pose *= [1e-3, 1e-3, 1e-3, np.pi/180, np.pi/180, np.pi/180]
        move *= [1e-3, 1e-3, 1e-3, np.pi/180, np.pi/180, np.pi/180]

        # pose is relative to object
        pose += obj_pose

        # move to slightly above new pose (avoid changing pose in contact with object)
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

    setup_data_collection = {
        "surface_3d": setup_surface_3d_data_collection,
        "edge_2d": setup_edge_2d_data_collection,
        "edge_3d": setup_edge_3d_data_collection,
        "edge_5d": setup_edge_5d_data_collection
    }

    num_samples = 10
    collect_dir_name = "example_data"
    
    for task in tasks:

        target_df, image_dir = setup_data_collection[task](
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
