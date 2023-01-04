def setup_surface_3d_servo_control():

    env_params_list = [{
        'stim_name': 'saddle', 
        'stim_pose': [600, 0, 12.5, 0, 0, 0],
        'workframe': [600, -40, 65, -180, 0, 90], 
    }]
    
    control_params = {
        'ep_len':100,
        'ref_pose': [1, 0, 3, 0, 0, 0],
        'p_gains': [1, 1, 0.5, 0.5, 0.5, 1],
        'i_gains': [0, 0, 0.3, 0.1, 0.1, 0],
        'i_clip': [[0, 0, 0, -30, -30, 0], [0, 0, 5, 30, 30, 0]]
    }

    return env_params_list, control_params


def setup_edge_2d_servo_control():

    env_params_list = [
    {
        'stim_name': 'square', 
        'stim_pose': [600, 0, 12.5, 0, 0, 0],
        'workframe': [650, 0, 52.5 - 3, -180, 0, 90] 
    },
    {
        'stim_name': 'circle', 
        'stim_pose': [600, 0, 12.5, 0, 0, 0],
        'workframe': [650, 0, 52.5 - 3, -180, 0, 90]
    },
    {
        'stim_name': 'clover', 
        'stim_pose': [600, 0, 12.5, 0, 0, 0],
        'workframe': [650, 0, 52.5 - 3, -180, 0, 90]
    },
    {
        'stim_name': 'foil', 
        'stim_pose': [600, 0, 12.5, 0, 0, 0],
        'workframe': [650, -10, 52.5 - 3, -180, 0, 90]
    }
    ]

    control_params = {
        'ep_len': 350,
        'ref_pose': [1, 0, 0, 0, 0, 0],
        'p_gains': [1, 0.5, 0, 0, 0, 0.5],
        'i_gains': [0, 0.3, 0, 0, 0, 0.1],
        'i_clip': [[0, -5, 0, 0, 0, -45], [0, 5, 0, 0, 0, 45]]
    }

    return env_params_list, control_params


def setup_edge_3d_servo_control():

    env_params_list = [{
        'stim_name': 'saddle', 
        'stim_pose': [600, 0, 12.5, 0, 0, 0],
        'workframe': [600, -65, 70, -180, 0, 0], 
    }]

    control_params = {
        'ep_len': 400,
        'ref_pose': [1, 0, 3, 0, 0, 0] ,
        'p_gains': [1, 0.5, 0.5, 0, 0, 0.5],
        'i_gains': [0, 0.3, 0.3, 0, 0, 0.1],
        'i_clip':[[0, -5, 0, 0, 0, -45], [0, 5, 5, 0, 0, 45]]
    }

    return env_params_list, control_params


def setup_edge_5d_servo_control():

    env_params_list = [{
        'stim_name': 'saddle', 
        'stim_pose': [600, 0, 12.5, 0, 0, 0],
        'workframe': [600, -65, 70, -180, 0, 0], 
    }]

    control_params = {
        'ep_len': 500,
        'ref_pose': [1, 0, 4, 0, 0, 0],
        'p_gains': [1, 0.5, 0.5, 0.5, 0.5, 0.5],
        'i_gains': [0, 0.3, 0.3, 0.1, 0.1, 0.1],
        'i_clip': [[0, -5, 0, -30, -30, -45], [0, 5, 5, 30, 30, 45]]
    }

    return env_params_list, control_params


setup_servo_control = {
    "surface_3d": setup_surface_3d_servo_control,
    "edge_2d": setup_edge_2d_servo_control,
    "edge_3d": setup_edge_3d_servo_control,
    "edge_5d": setup_edge_5d_servo_control
}


if __name__ == '__main__':
    pass
