from cri.transforms import transform_pose

def setup_stim(stimulus, task):

    stim_pose = [450, -100, 40, 0, 0, 0]
    work_frame = {
        'pushing_2d': [400, 0, 52.5-3, -180, 0, 90],
        'pushing_3d': [stim_pose[0], stim_pose[1]-40.0, stim_pose[2], -180, 0, 90]
    }

    stim_params_dict = {
        'cube_'+task: {
            'stim_name': 'cube', 
            'stim_pose': stim_pose,
            'stim_fixed': False,
            'work_frame': work_frame[task],
            'target_pose': [400+200, 300, 0,  0, 0, 0]
        },
        'circle_'+task: {
            'stim_name': 'circle', 
            'stim_pose': stim_pose,
            'stim_fixed': False,
            'work_frame': work_frame[task],
            'target_pose': [400+200, 300, 0,  0, 0, 00]
        },
    }

    stim_params = stim_params_dict[stimulus + '_' + task]

    return stim_params


def setup_3d_push(stimulus):

    env_params = setup_stim(stimulus, 'pushing_3d')

    cam_params = {
        'image_size': [512, 512],
        'dist': 1.2,
        'yaw': 90.0,
        'pitch': -45.0,
        'pos': [0.1, 0.0, -0.35],
        'fov': 75.0,
        'near_val': 0.1,
        'far_val': 100.0
    }

    env_params.update({
        'cam_params': cam_params,
        'quick_mode': False
    })

    control_params = {
        'ep_len': 1000,
        'pid_pose_params': {
            'kp': [0.9, 0, 0, 0, 0, 0.9],                  # irregular objects / flat surface
            'ki': [0.1, 0, 0, 0, 0, 0.1],                  # irregular objects / flat surface
            'ei_clip': [[-5, -5, -5, -25, -25, -25], [5, 5, 5, 25, 25, 25]],   # integral error clipping
            'alpha': 0.5,                                          # differential error filtering coeff
            'error': lambda y, r: transform_pose(r, y),            # SE(3) error
        },
        'ref_pose': [1.5, 0, 0, 0, 0, 0],                          # flat surface ref pose
        'pid_align_params': {
            'kp': 0.2,                                             # proportional gain
            'kd': 0.9,                                             # differential gain
            'ep_clip': [-10, 10],                                  # proportional error clipping
            'alpha': 0.5,                                          # differential error filtering coeff
        },
        'ref_align': 0,                                            # target bearing (in servo frame)        
        'target_pose': env_params['target_pose']
    }

    return env_params, control_params


setup_tactile_pushing = {
    "pushing_3d": setup_3d_push,
}


if __name__ == '__main__':
    pass
