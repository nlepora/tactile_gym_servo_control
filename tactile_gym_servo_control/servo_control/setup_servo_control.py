from tactile_gym_servo_control.utils.load_embodiment_and_env import POSE_UNITS


def setup_surface_3d_servo_control():

    setup_servo_control = {
        "saddle": [-40, 0, -10, 0, 0, 0] * POSE_UNITS, 
    }

    stim_names = list(setup_servo_control.keys())
    ep_len = 200
    init_poses = list(setup_servo_control.values())
    ref_pose = [1, 0, 3, 0, 0, 0] 
    p_gains = [1, 1, 0.5, 0.5, 0.5, 1]

    return stim_names, ep_len, init_poses, ref_pose, p_gains


def setup_edge_2d_servo_control():

    setup_servo_control = {
        "square": [0, -50, 4, 0, 0, 0] * POSE_UNITS,
        "circle": [0, -50, 4, 0, 0, 0] * POSE_UNITS,
        "clover": [0, -50, 4, 0, 0, 0] * POSE_UNITS,
        "foil":   [0, -40, 4, 0, 0, 0] * POSE_UNITS,
    }

    stim_names = list(setup_servo_control.keys())
    ep_len = 350
    init_poses = list(setup_servo_control.values())
    ref_pose = [1, 0, 0, 0, 0, 0] 
    p_gains = [1, 0.5, 0, 0, 0, 0.5]

    return stim_names, ep_len, init_poses, ref_pose, p_gains


def setup_edge_3d_servo_control():

    setup_servo_control = {
        "saddle": [-70, 0, -20, 0, 0, -90] * POSE_UNITS, 
    }

    stim_names = list(setup_servo_control.keys())
    ep_len = 500
    init_poses = list(setup_servo_control.values())
    ref_pose = [1, 0, 3, 0, 0, 0] 
    p_gains = [1, 0.5, 0.5, 0, 0, 0.5]

    return stim_names, ep_len, init_poses, ref_pose, p_gains


def setup_edge_5d_servo_control():

    setup_servo_control = {
        "saddle": [-70, 0, -20, 0, 0, -90] * POSE_UNITS, 
    }

    stim_names = list(setup_servo_control.keys())
    ep_len = 500
    init_poses = list(setup_servo_control.values())
    ref_pose = [1, 0, 4, 0, 0, 0] 
    p_gains = [1, 0.5, 0.5, 0.5, 0.5, 0.5]

    return stim_names, ep_len, init_poses, ref_pose, p_gains


setup_servo_control = {
    "surface_3d": setup_surface_3d_servo_control,
    "edge_2d": setup_edge_2d_servo_control,
    "edge_3d": setup_edge_3d_servo_control,
    "edge_5d": setup_edge_5d_servo_control
}


if __name__ == '__main__':
    pass
