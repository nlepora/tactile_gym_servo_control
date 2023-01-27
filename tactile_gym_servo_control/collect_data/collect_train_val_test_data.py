"""
python collect_train_val_test_data.py -t surface_3d
python collect_train_val_test_data.py -t edge_2d
python collect_train_val_test_data.py -t edge_3d
python collect_train_val_test_data.py -t edge_5d
python collect_train_val_test_data.py -t surface_3d edge_2d edge_3d edge_5d
"""

import os
import argparse

from tactile_gym_servo_control.utils.setup_embodiment_sim import setup_embodiment
from tactile_gym_servo_control.collect_data.setup_collect_sim_data import setup_collect_data
from tactile_gym_servo_control.collect_data.collect_data import collect_data

data_path = os.path.join(os.path.dirname(__file__), '../../example_data/sim')
data_version = ''


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--tasks',
        nargs='+',
        help="Choose task from [surface_3d edge_2d edge_3d edge_5d].",
        default=['edge_5d']
    )

    # parse arguments
    args = parser.parse_args()
    tasks = args.tasks
    
    collection_params = {
        'train': 5000,
        'val': 2000,
        # 'test': 2000
    }

    for task in tasks:

        for collect_dir_name, num_samples in collection_params.items():

            collect_dir = os.path.join(
                data_path, task + data_version, collect_dir_name
            )

            target_df, image_dir, env_params, sensor_params = \
                setup_collect_data[task](
                    collect_dir, num_samples
                )

            embodiment = setup_embodiment(
                **env_params, 
                sensor_params = sensor_params,
                show_gui=True, quick_mode=True 
            )

            collect_data(
                embodiment,
                target_df,
                image_dir
            )
