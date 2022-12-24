"""
python collect_train_and_val_sets.py -t surface_3d
python collect_train_and_val_sets.py -t edge_2d
python collect_train_and_val_sets.py -t edge_3d
python collect_train_and_val_sets.py -t edge_5d
python collect_train_and_val_sets.py -t surface_3d edge_2d edge_3d edge_5d
"""

import os
import argparse

from tactile_gym_servo_control.utils.pybullet_utils import setup_pybullet_env
from tactile_gym_servo_control.data_collection.collect_data import load_embodiment_and_env
from tactile_gym_servo_control.data_collection.collect_data import collect_data
from tactile_gym_servo_control.data_collection.setup_data_collection import setup_surface_3d_data_collection
from tactile_gym_servo_control.data_collection.setup_data_collection import setup_edge_2d_data_collection
from tactile_gym_servo_control.data_collection.setup_data_collection import setup_edge_3d_data_collection
from tactile_gym_servo_control.data_collection.setup_data_collection import setup_edge_5d_data_collection

stimuli_path = os.path.join(os.path.dirname(__file__), "../stimuli")


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

    show_tactile = False
    quick_mode = True

    collection_params = {
        'train': 5000,
        'val': 2000
    }

    for task in tasks:

        for collect_dir_name, num_samples in collection_params.items():

            target_df, image_dir, workframe_pos, workframe_rpy = setup_data_collection[task](
                num_samples=num_samples,
                shuffle_data=False,
                collect_dir_name=collect_dir_name,
            )

            embodiment = load_embodiment_and_env(
                show_tactile=show_tactile
            )

            collect_data(
                embodiment,
                target_df,
                image_dir,
                quick_mode=quick_mode
            )
