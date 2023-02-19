from cri.transforms import transform_pose

def setup_stim(stimulus, task):

    stim_pose = [600, 0, 0, 0, 0, 0]
    work_frame = {
        'edge':    [650,   0, 40-3, -180, 0, 0],
        'surface': [610, -55, 40-3, -180, 0, 0],
        'saddle':  [600, -65, 55-3, -180, 0, 0]
    }

    stim_params_dict = {
        'saddle_surface': {
            'stim_name': 'saddle',
            'stim_pose': stim_pose,
            'work_frame': work_frame['saddle'], 
        },
        'saddle_edge': {
            'stim_name': 'saddle',
            'stim_pose': stim_pose,
            'work_frame': work_frame['saddle'], 
        },
        'bowl_surface': {
            'stim_name': 'bowl',
            'stim_pose': [600, 0, 60, 90, 0, 0],
            'stim_scale': 0.3,
            'work_frame': [600, 0, 25, -180, 0, 0], 
        },
        'square_'+task: {
            'stim_name': 'square', 
            'stim_pose': stim_pose,
            'work_frame': work_frame[task]
        },
        'circle_'+task: {
            'stim_name': 'circle', 
            'stim_pose': stim_pose,
            'work_frame': work_frame[task] 
        },
        'clover_'+task: {
            'stim_name': 'clover', 
            'stim_pose': stim_pose,
            'work_frame': work_frame[task] 
        },
        'foil_'+task: {
            'stim_name': 'foil', 
            'stim_pose': stim_pose,
            'work_frame': work_frame[task]
        }
    }

    env_params = stim_params_dict[stimulus + '_' + task]

    env_params.update({
        'show_gui': True, 
        'show_tactile': True, 
        'quick_mode': False
    })

    return env_params


def setup_surface_3d_servo(stimulus):

    env_params = setup_stim(stimulus, 'surface')

    control_params = {
        'ep_len': 500,
        'pid_params': {
            'kp': [1, 1, 0.5, 0.5, 0.5, 1],                 
            'ki': [0, 0, 0.3, 0.1, 0.1, 0],                
            'ei_clip': [[0, 0, 0, -30, -30, 0], [0, 0, 5, 30, 30, 0]],        
            'error': lambda y, r: transform_pose(r, y)  # SE(3) error
        },
        'ref_pose': [0, -1, 3, 0, 0, 0]              
    }

    return env_params, control_params


def setup_edge_2d_servo(stimulus):

    env_params = setup_stim(stimulus, 'edge')

    control_params = {
        'ep_len': 100,#60,
        'pid_params': {
            'kp': [0.5, 1, 0, 0, 0, 0.5],                 
            'ki': [0.3, 0, 0, 0, 0, 0.1],                
            'ei_clip': [[-5, 0, 0, 0, 0, -45], [5, 0, 0, 0, 0,  45]],          
            'error': lambda y, r: transform_pose(r, y)  # SE(3) error
        },
        'ref_pose': [0, 1, 0, 0, 0, 0],              
    }

    return env_params, control_params


def setup_edge_3d_servo(stimulus):

    env_params = setup_stim(stimulus, 'edge')

    control_params = {
        'ep_len': 400,
        'pid_params': {
            'kp': [0.5, 1, 0.5, 0, 0, 0.5],                 
            'ki': [0.3, 0, 0.3, 0, 0, 0.1],                
            'ei_clip': [[0, -5, 0, 0, 0, -45], [0, 5, 5, 0, 0, 45]],
            'error': lambda y, r: transform_pose(r, y)  # SE(3) error
        },
        'ref_pose': [0, 1, 3, 0, 0, 0],              
    }

    return env_params, control_params


def setup_edge_5d_servo(stimulus):

    env_params = setup_stim(stimulus, 'edge')

    control_params = {
        'ep_len': 250,
        'pid_params': {
            'kp': [1, 0.5, 0.5, 0.5, 0.5, 0.5],                
            'ki': [0, 0.3, 0.3, 0.1, 0.1, 0.1],
            'ei_clip': [[0, -5, 0, -30, -30, -45], [0, 5, 5, 30, 30, 45]],
            'error': lambda y, r: transform_pose(r, y)  # SE(3) error
        },
        'ref_pose': [1, 0, 3, 0, 0, 0],              
    }

    return env_params, control_params


setup_servo = {
    "surface_3d": setup_surface_3d_servo,
    "edge_2d": setup_edge_2d_servo,
    "edge_3d": setup_edge_3d_servo,
    "edge_5d": setup_edge_5d_servo
}


if __name__ == '__main__':
    pass
