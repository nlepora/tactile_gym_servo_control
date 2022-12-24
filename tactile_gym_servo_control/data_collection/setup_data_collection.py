import os
import numpy as np

from tactile_gym.utils.general_utils import save_json_obj

from tactile_gym_servo_control.data_collection.data_collection_utils import make_target_df_rand
from tactile_gym_servo_control.data_collection.data_collection_utils import create_data_dir


def setup_surface_3d_data_collection(
    num_samples=100,
    shuffle_data=False,
    collect_dir_name=None,
):

    obj_poses = [[0, 0, 0, 0, 0, 0]]
    poses_rng = [
        [0.0, 0.0, 0.5, -25.0, -25.0, 0.0],
        [0.0, 0.0, 5.5,  25.0,  25.0, 0.0]
    ]
    moves_rng = [
        [0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0]
    ]

    target_df = make_target_df_rand(
        poses_rng, moves_rng, num_samples, obj_poses, shuffle_data
    )

    collect_dir, image_dir = create_data_dir(
        target_df,
        "surface_3d",
        collect_dir_name=collect_dir_name,
    )

    collection_params = {
        'obj_poses': obj_poses,
        'poses_rng': poses_rng,
        'moves_rng': moves_rng,
    }

    save_json_obj(collection_params, os.path.join(collect_dir, 'collection_params'))

    return target_df, image_dir


def setup_edge_2d_data_collection(
    num_samples=100,
    shuffle_data=False,
    collect_dir_name=None,
):

    obj_poses = [[0, 0, 0, 0, 0, 0]]
    poses_rng = [
        [0.0, -4.0, 2.0, -2.5, -2.5, -180.0],
        [0.0,  4.0, 5.5,  2.5,  2.5,  180.0]
    ]
    moves_rng = [
        [0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0]
    ]

    target_df = make_target_df_rand(
        poses_rng, moves_rng, num_samples, obj_poses, shuffle_data
    )

    collect_dir, image_dir = create_data_dir(
        target_df,
        "edge_2d",
        collect_dir_name=collect_dir_name,
    )

    collection_params = {
        'obj_poses': obj_poses,
        'poses_rng': poses_rng,
        'moves_rng': moves_rng,
    }

    save_json_obj(collection_params, os.path.join(collect_dir, 'collection_params'))

    return target_df, image_dir


def setup_edge_3d_data_collection(
    num_samples=100,
    shuffle_data=False,
    collect_dir_name=None,
):

    obj_poses = [[0, 0, 0, 0, 0, 0]]
    poses_rng = [
        [0.0, -3.0, 2.0, -20.0, -20.0, -180.0],
        [0.0,  3.0, 5.5,  20.0,  20.0,  180.0]
    ]
    moves_rng = [
        [0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0]
    ]

    target_df = make_target_df_rand(
        poses_rng, moves_rng, num_samples, obj_poses, shuffle_data
    )

    collect_dir, image_dir = create_data_dir(
        target_df,
        "edge_3d",
        collect_dir_name=collect_dir_name,
    )

    collection_params = {
        'obj_poses': obj_poses,
        'poses_rng': poses_rng,
        'moves_rng': moves_rng,
    }

    save_json_obj(collection_params, os.path.join(collect_dir, 'collection_params'))

    return target_df, image_dir


def setup_edge_5d_data_collection(
    num_samples=100,
    shuffle_data=False,
    collect_dir_name=None,
):

    obj_poses = [[0, 0, 0, 0, 0, 0]]
    poses_rng = [
        [0.0, -4.0, 2.0, -15.0, -15.0, -180.0],
        [0.0,  4.0, 5.5,  15.0,  15.0,  180.0]
    ]
    moves_rng = [
        [0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0]
    ]
    
    target_df = make_target_df_rand(
        poses_rng, moves_rng, num_samples, obj_poses, shuffle_data
    )

    collect_dir, image_dir = create_data_dir(
        target_df,
        "edge_5d",
        collect_dir_name=collect_dir_name,
    )

    collection_params = {
        'obj_poses': obj_poses,
        'poses_rng': poses_rng,
        'moves_rng': moves_rng,
    }

    save_json_obj(collection_params, os.path.join(collect_dir, 'collection_params'))

    return target_df, image_dir