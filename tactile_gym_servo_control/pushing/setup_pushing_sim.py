def setup_stim(stimulus, task):

    stim_pose = [450, -100, 40.0, 0, 0, 0]
    workframe = {
        'pushing_2d':   [400,   0, 52.5 - 3, -180, 0, 90],
        'pushing_3d':   [stim_pose[0], stim_pose[1]-40.0, stim_pose[2], -180, 0, 90]
    }

    stim_params_dict = {
        'cube_'+task: {
            'stim_name': 'cube', 
            'stim_pose': stim_pose,
            'workframe': workframe[task],
            'fix_stim': False,
            'hover': [-7.5, 0.0, 0, 0, 0, 0],
            'work_target_pose': [200, 300, 0,  0, 0, 0],
            
        },
        'circle_'+task: {
            'stim_name': 'circle', 
            'stim_pose': stim_pose,
            'workframe': workframe[task],
            'fix_stim': False,
            'hover': [-7.5, 0.0, 0, 0, 0, 0],
            'work_target_pose': [400, 200, 0,  0, 0, 0],
        },
    }

    stim_params = stim_params_dict[stimulus + '_' + task]

    return stim_params


def setup_3d_push(stimulus):

    env_params = setup_stim(stimulus, 'pushing_3d')

    control_params = {
        'ep_len': 1000,
        'ref_pose': [0, -2, 3, 0, 0, 0],
        'p_gains': [1, 1, 0.5, 0.5, 0.5, 1],
        'i_gains': [0, 0, 0.3, 0.1, 0.1, 0],
        'i_clip': [[0, 0, 0, -30, -30, 0], [0, 0, 5, 30, 30, 0]],
        'work_target_pose': env_params['work_target_pose'],
    }

    return env_params, control_params


setup_tactile_pushing = {
    "pushing_3d": setup_3d_push,
}


if __name__ == '__main__':
    pass
