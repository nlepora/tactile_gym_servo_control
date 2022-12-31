import os

from tactile_gym.utils.general_utils import save_json_obj

from tactile_gym_servo_control.collect_data.utils_collect_data import make_target_df_rand
from tactile_gym_servo_control.collect_data.utils_collect_data import create_data_dir

data_path = os.path.join(os.path.dirname(__file__), '../../example_data/real')


def setup_tactip(
    collect_dir
):
    tactip_params = {
        "size": [256, 256],
        "crop": [320-128-2, 240-128+20, 320+128-2, 240+128+20],
        "exposure": -7,
        "source": 0,
        "threshold": [61, -5],
        }

    save_json_obj(tactip_params, os.path.join(collect_dir, 'tactip_params'))

    return tactip_params


def setup_edge_2d_collect_data(
    num_samples=10,
    collect_dir_name="data",
    shuffle_data=False,
):
    env_params = {
        'workframe': [285, 0, -93, 0, 0, -90],
        'linear_speed': 10, 
        'angular_speed': 10,
        'tcp_pose': [0, 0, 0, 0, 0, 0]
    }

    pose_limits = {
        'pose_llims': [0, -4, 2, 0, 0, -180],
        'pose_ulims': [0,  4, 3, 0, 0,  180],
        'obj_poses':[[0, 0, 0, 0, 0, 0]]
    }

    collect_dir = os.path.join(
        data_path, 'edge_2d', collect_dir_name
    )

    target_df = make_target_df_rand(
        num_samples, shuffle_data, **pose_limits
    )

    image_dir = create_data_dir(collect_dir, target_df)

    tactip_params = setup_tactip(collect_dir)

    save_json_obj(pose_limits, os.path.join(collect_dir, 'pose_limits'))
    save_json_obj(env_params, os.path.join(collect_dir, 'env_params'))

    return target_df, image_dir, env_params, tactip_params


def setup_edge_3d_collect_data(
    num_samples=10,
    collect_dir_name="data",
    shuffle_data=False,
):
    env_params = {
        'workframe': [285, 0, -93, 0, 0, -90],
        'linear_speed': 10, 
        'angular_speed': 10,
        'tcp_pose': [0, 0, 0, 0, 0, 0]
    }

    pose_limits = {
        'pose_llims': [0, -4, 0.5, 0, 0, -180],
        'pose_ulims': [0,  4, 5, 0, 0,  180],
        'obj_poses':[[0, 0, 0, 0, 0, 0]]
    }

    collect_dir = os.path.join(
        data_path, 'edge_3d', collect_dir_name
    )

    target_df = make_target_df_rand(
        num_samples, shuffle_data, **pose_limits
    )

    image_dir = create_data_dir(collect_dir, target_df)

    tactip_params = setup_tactip(collect_dir)

    save_json_obj(pose_limits, os.path.join(collect_dir, 'pose_limits'))
    save_json_obj(env_params, os.path.join(collect_dir, 'env_params'))

    return target_df, image_dir, env_params, tactip_params


setup_collect_data = {
    "edge_2d": setup_edge_2d_collect_data,
    "edge_3d": setup_edge_3d_collect_data
}


if __name__ == '__main__':
    pass
