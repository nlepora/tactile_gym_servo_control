"""
python collect_train_val_test_data.py -t surface_3d
python collect_train_val_test_data.py -t edge_2d
python collect_train_val_test_data.py -t edge_3d
python collect_train_val_test_data.py -t edge_5d
python collect_train_val_test_data.py -t surface_3d edge_2d edge_3d edge_5d
"""

import argparse

from tactile_gym_servo_control.utils_robot_real.setup_embodiment_env import setup_embodiment_env
from tactile_gym_servo_control.collect_data.setup_collect_real_data import setup_collect_data
from tactile_gym_servo_control.collect_data.collect_data import collect_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--tasks',
        nargs='+',
        help="Choose task from ['edge_2d', 'edge_3d'].",
        default=['edge_3d']
    )

    # parse arguments
    args = parser.parse_args()
    tasks = args.tasks

    collection_params = {
        'train': 2000,
        'val': 1000
    }

    for task in tasks:

        for collect_dir_name, num_samples in collection_params.items():

            target_df, image_dir, env_params, tactip_params = \
                setup_collect_data[task](
                    num_samples=num_samples,
                    collect_dir_name=collect_dir_name,
                )

            embodiment = setup_embodiment_env(
                **env_params, 
                tactip_params = tactip_params,  
            )

            collect_data(
                embodiment,
                target_df,
                image_dir
            )
