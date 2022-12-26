import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def setup_surface_3d_servo_control():

    setup_servo_control = {
        "saddle": [-0.04, 0.0, -0.01, 0.0, 0.0, 0.0], # [m, m, m, rad, rad, rad]
    }

    stim_names = list(setup_servo_control.keys())
    ep_len = 200
    init_poses = list(setup_servo_control.values())
    ref_pose = [1.0, 0.0, 2.5, 0.0, 0.0, 0.0] # [mm, mm, mm, deg, deg, deg]
    p_gains = [0.5, 0.5, 0.5, 0.1, 0.1, 0.0]

    return stim_names, ep_len, init_poses, ref_pose, p_gains


def setup_edge_2d_servo_control():

    setup_servo_control = {
        "square": [0.0, -0.05, 0.004, 0.0, 0.0, 0.0], # [m, m, m, rad, rad, rad]
        "circle": [0.0, -0.05, 0.004, 0.0, 0.0, 0.0],
        "clover": [0.0, -0.05, 0.004, 0.0, 0.0, 0.0],
        "foil":   [0.0, -0.04, 0.004, 0.0, 0.0, 0.0],
    }

    stim_names = list(setup_servo_control.keys())
    ep_len = 350
    init_poses = list(setup_servo_control.values())
    ref_pose = [1.0, 0.0, 2.5, 0.0, 0.0, 0.0] # [mm, mm, mm, deg, deg, deg]
    p_gains = [1.0, 1.0, 0.0, 0.0, 0.0, 0.1]

    return stim_names, init_poses, ep_len, ref_pose, p_gains


def setup_edge_3d_servo_control():

    setup_servo_control = {
        "saddle": [-0.07, 0.0, -0.02, 0.0, 0.0, -np.pi/2], # [m, m, m, rad, rad, rad]
    }

    stim_names = list(setup_servo_control.keys())
    ep_len = 500
    init_poses = list(setup_servo_control.values())
    ref_pose = [1.0, 0.0, 3.5, 0.0, 0.0, 0.0] # [mm, mm, mm, deg, deg, deg]
    p_gains = [1.0, 1.0, 0.5, 0.0, 0.0, 0.05]

    return stim_names, ep_len, init_poses, ref_pose, p_gains


def setup_edge_5d_servo_control():

    setup_servo_control = {
        "saddle": [-0.07, 0.0, -0.02, 0.0, 0.0, -np.pi/2], # [m, m, m, rad, rad, rad]
    }

    stim_names = list(setup_servo_control.keys())
    ep_len = 500
    init_poses = list(setup_servo_control.values())
    ref_pose = [1.0, 0.0, 4.0, 0.0, 0.0, 0.0] # [mm, mm, mm, deg, deg, deg]
    p_gains = np.array([1.0, 1.0, 0.5, 0.05, 0.05, 0.05])

    return stim_names, ep_len, init_poses, ref_pose, p_gains


if __name__ == '__main__':
    pass
