def setup_stim(stimulus, task):

    stim_pose = [600, 0, 0, 0, 0, 0]
    workframe = {
        'edge':    [650,   0, 40-3, -180, 0, 0],
        'surface': [610, -55, 40-3, -180, 0, 0],
        'saddle': [600, -65, 55-3, -180, 0, 0]
    }

    stim_params_dict = {
        'saddle_surface': {
            'stim_name': 'saddle',
            'stim_pose': stim_pose,
            'workframe': workframe['saddle'], 
        },
        'saddle_edge': {
            'stim_name': 'saddle',
            'stim_pose': stim_pose,
            'workframe': workframe['saddle'], 
        },
        'bowl_surface': {
            'stim_name': 'bowl',
            'stim_pose': [600, 0, 60, 90, 0, 0],
            'stim_scale': 0.3,
            'workframe': [600, 0, 25, -180, 0, 0], 
        },
        'square_'+task: {
            'stim_name': 'square', 
            'stim_pose': stim_pose,
            'workframe': workframe[task]
        },
        'circle_'+task: {
            'stim_name': 'circle', 
            'stim_pose': stim_pose,
            'workframe': workframe[task] 
        },
        'clover_'+task: {
            'stim_name': 'clover', 
            'stim_pose': stim_pose,
            'workframe': workframe[task] 
        },
        'foil_'+task: {
            'stim_name': 'foil', 
            'stim_pose': stim_pose,
            'workframe': workframe[task]
        }
    }

    stim_params = stim_params_dict[stimulus + '_' + task]

    return stim_params


def setup_surface_3d_servo_control(stimulus):

    env_params = setup_stim(stimulus, 'surface')

    control_params = {
        'ep_len': 500,
        'ref_pose': [0, -1, 3, 0, 0, 0],
        'p_gains': [1, 1, 0.5, 0.5, 0.5, 1],
        'i_gains': [0, 0, 0.3, 0.1, 0.1, 0],
        'i_clip': [[0, 0, 0, -30, -30, 0], [0, 0, 5, 30, 30, 0]]
    }

    return env_params, control_params


def setup_edge_2d_servo_control(stimulus):

    env_params = setup_stim(stimulus, 'edge')

    control_params = {
        'ep_len': 160,
        'ref_pose': [0, 1, 0, 0, 0, 0],
        'p_gains': [0.5, 1, 0, 0, 0, 0.5],
        'i_gains': [0.3, 1, 0, 0, 0, 0.1],
        'i_clip': [[-5, 0, 0, 0, 0, -45], [5, 0, 0, 0, 0, 45]]
    }

    return env_params, control_params


def setup_edge_3d_servo_control(stimulus):

    env_params = setup_stim(stimulus, 'edge')

    control_params = {
        'ep_len': 400,
        'ref_pose': [0, 1, 3, 0, 0, 0] ,
        'p_gains': [0.5, 1, 0.5, 0, 0, 0.5],
        'i_gains': [0, 0.3, 0.3, 0, 0, 0.1],
        'i_clip':[[0, -5, 0, 0, 0, -45], [0, 5, 5, 0, 0, 45]]
    }

    return env_params, control_params


def setup_edge_5d_servo_control(stimulus):

    env_params = setup_stim(stimulus, 'edge')

    control_params = {
        'ep_len': 250,
        'ref_pose': [0, 1, 3, 0, 0, 0],
        'p_gains': [0.5, 1, 0.5, 0.5, 0.5, 0.5],
        'i_gains': [0, 0.3, 0.3, 0.1, 0.1, 0.1],
        'i_clip': [[0, -5, 0, -30, -30, -45], [0, 5, 5, 30, 30, 45]]
    }

    return env_params, control_params


setup_servo_control = {
    "surface_3d": setup_surface_3d_servo_control,
    "edge_2d": setup_edge_2d_servo_control,
    "edge_3d": setup_edge_3d_servo_control,
    "edge_5d": setup_edge_5d_servo_control
}


if __name__ == '__main__':
    pass
