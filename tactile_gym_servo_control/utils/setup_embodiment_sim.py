import os
import cv2
import numpy as np

from cri.robot import SyncRobot
from cri.controller import SimController as Controller
from tactile_gym_servo_control.utils.image_transforms import process_image

stimuli_path = os.path.join(os.path.dirname(__file__), 'stimuli')


def setup_embodiment(
    env_params={},
    sensor_params={},
    hover=[0, 0, -7.5, 0, 0, 0]
):
    workframe = env_params.get('workframe', [600, 0, 0, 0, 0, 0])
    tcp_pose = env_params.get('tcp_pose', [0, 0, 0, 0, 0, 0])
    env_params['stim_path'] = stimuli_path

    # setup the embodiment
    embodiment = SyncRobot(Controller(sensor_params, env_params))   
    sensor = embodiment.controller._client._sim_env

    # setup the tactile sensor
    def sensor_process(outfile=None):
        img = sensor.get_tactile_observation()
        img = process_image(img, **sensor_params)
        if outfile is not None:
            cv2.imwrite(outfile, img)
        return img

    embodiment.sensor_process = sensor_process

    # settings
    embodiment.coord_frame = workframe
    embodiment.tcp = tcp_pose
    embodiment.hover = np.array(hover)

    return embodiment


if __name__ == '__main__':
    pass
