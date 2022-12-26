import os
import numpy as np

from tactile_gym.utils.general_utils import save_json_obj

from tactile_gym_servo_control.data_collection.data_collection_utils import make_target_df_rand
from tactile_gym_servo_control.data_collection.data_collection_utils import create_data_dir

data_path = os.path.join(os.path.dirname(__file__), '../../example_data')


def setup_surface_3d_data_collection(
    num_samples=10,
    collect_dir_name="data",
    shuffle_data=False,
):

    pose_limits = {
        'pose_llims': [0, 0, 0.5, -25, -25, 0],
        'pose_ulims': [0, 0, 5.5,  25,  25, 0],
        'obj_poses':[[0, 0, 0, 0, 0, 0]]
    }

    collect_dir = os.path.join(
        data_path, 'surface_3d', collect_dir_name
    )

    target_df = make_target_df_rand(
        num_samples, shuffle_data, **pose_limits
    )

    image_dir = create_data_dir(collect_dir, target_df)

    save_json_obj(pose_limits, os.path.join(collect_dir, 'pose_limits'))

    return target_df, image_dir


def setup_edge_2d_data_collection(
    num_samples=10,
    collect_dir_name="data",
    shuffle_data=False,
):

    pose_limits = {
        'pose_llims': [0, -4, 2,  -2.5, -2.5, -180],
        'pose_ulims': [0,  4, 5.5, 2.5,  2.5,  180],
        'obj_poses':[[0, 0, 0, 0, 0, 0]]
    }

    collect_dir = os.path.join(
        data_path, 'surface_3d', collect_dir_name
    )

    target_df = make_target_df_rand(
        num_samples, shuffle_data, **pose_limits
    )

    image_dir = create_data_dir(collect_dir, target_df)

    save_json_obj(pose_limits, os.path.join(collect_dir, 'pose_limits'))

    return target_df, image_dir


def setup_edge_3d_data_collection(
    num_samples=10,
    collect_dir_name="data",
    shuffle_data=False,
):

    pose_limits = {
        'pose_llims': [0, -3, 2,  -20, -20, -180],
        'pose_ulims': [0,  3, 5.5, 20,  20,  180],
        'obj_poses':[[0, 0, 0, 0, 0, 0]]
    }

    collect_dir = os.path.join(
        data_path, 'surface_3d', collect_dir_name
    )

    target_df = make_target_df_rand(
        num_samples, shuffle_data, **pose_limits
    )

    image_dir = create_data_dir(collect_dir, target_df)

    save_json_obj(pose_limits, os.path.join(collect_dir, 'pose_limits'))

    return target_df, image_dir


def setup_edge_5d_data_collection(
    num_samples=10,
    collect_dir_name="data",
    shuffle_data=False,
):

    pose_limits = {
        'pose_llims': [0, -4,   2, -15, -15, -180],
        'pose_ulims': [0,  4, 5.5,  15,  15,  180],
        'obj_poses':[[0, 0, 0, 0, 0, 0]]
    }

    collect_dir = os.path.join(
        data_path, 'surface_3d', collect_dir_name
    )

    target_df = make_target_df_rand(
        num_samples, shuffle_data, **pose_limits
    )

    image_dir = create_data_dir(collect_dir, target_df)

    save_json_obj(pose_limits, os.path.join(collect_dir, 'pose_limits'))

    return target_df, image_dir
