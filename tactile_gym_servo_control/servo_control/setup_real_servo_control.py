def setup_edge_2d_servo_control():

    env_params_list = [
    {
        'workframe': [285, 0, -93+1.5, 0, 0, -90], # square, circle
        'linear_speed': 10, 
        'angular_speed': 10,
        'tcp_pose': [0, 0, 0, 0, 0, 0]    
    }]

    control_params = {
        'ep_len': 350,
        'ref_pose': [1, 0, 0, 0, 0, 0],
        'p_gains': [1, 0.5, 0, 0, 0, 0.5],
        'i_gains': [0, 0.3, 0, 0, 0, 0.1],
        'i_clip': [[0, -5, 0, 0, 0, -45], [0, 5, 0, 0, 0, 45]]
    }

    return env_params_list, control_params


setup_servo_control = {
    "edge_2d": setup_edge_2d_servo_control
}


if __name__ == '__main__':
    pass
