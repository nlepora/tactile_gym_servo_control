def setup_surface_3d_servo_control():

    setup_servo_control = {
        "saddle": [-40, 0, -10, 0, 0, 0], 
    }

    stim_names = list(setup_servo_control.keys())
    init_poses = list(setup_servo_control.values())
    ep_len = 100

    ref_pose = [1, 0, 3, 0, 0, 0]
    p_gains = [1, 1, 0.5, 0.5, 0.5, 1]
    i_gains = [0, 0, 0.3, 0.1, 0.1, 0]
    i_clip = [[0, 0, 0, -30, -30, 0], [0, 0, 5, 30, 30, 0]]

    return stim_names, init_poses, ep_len, ref_pose, p_gains, i_gains, i_clip


def setup_edge_2d_servo_control():

    setup_servo_control = {
        "square": [0, -50, 4, 0, 0, 0],
        "circle": [0, -50, 4, 0, 0, 0],
        "clover": [0, -50, 4, 0, 0, 0],
        "foil":   [0, -40, 4, 0, 0, 0],
    }

    stim_names = list(setup_servo_control.keys())
    init_poses = list(setup_servo_control.values())
    ep_len = 350

    ref_pose = [1, 0, 0, 0, 0, 0] 
    p_gains = [1, 0.5, 0, 0, 0, 0.5]
    i_gains = [0, 0.3, 0, 0, 0, 0.1]
    i_clip = [[0, -5, 0, 0, 0, -45], [0, 5, 0, 0, 0, 45]]

    return stim_names, init_poses, ep_len, ref_pose, p_gains, i_gains, i_clip


def setup_edge_3d_servo_control():

    setup_servo_control = {
        "saddle": [-70, 0, -20, 0, 0, -90], 
    }

    stim_names = list(setup_servo_control.keys())
    init_poses = list(setup_servo_control.values())
    ep_len = 400

    ref_pose = [1, 0, 3, 0, 0, 0] 
    p_gains = [1, 0.5, 0.5, 0, 0, 0.5]
    i_gains = [0, 0.3, 0.3, 0, 0, 0.1]
    i_clip = [[0, -5, 0, 0, 0, -45], [0, 5, 5, 0, 0, 45]]

    return stim_names, init_poses, ep_len, ref_pose, p_gains, i_gains, i_clip


def setup_edge_5d_servo_control():

    setup_servo_control = {
        "saddle": [-70, 0, -20, 0, 0, -90], 
    }

    stim_names = list(setup_servo_control.keys())
    init_poses = list(setup_servo_control.values())
    ep_len = 500

    ref_pose = [1, 0, 4, 0, 0, 0] 
    p_gains = [1, 0.5, 0.5, 0.5, 0.5, 0.5]
    i_gains = [0, 0.3, 0.3, 0.1, 0.1, 0.1]
    i_clip = [[0, -5, 0, -30, -30, -45], [0, 5, 5, 30, 30, 45]]

    return stim_names, init_poses, ep_len, ref_pose, p_gains, i_gains, i_clip


setup_servo_control = {
    "surface_3d": setup_surface_3d_servo_control,
    "edge_2d": setup_edge_2d_servo_control,
    "edge_3d": setup_edge_3d_servo_control,
    "edge_5d": setup_edge_5d_servo_control
}


if __name__ == '__main__':
    pass
