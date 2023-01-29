import os
import cv2
import numpy as np

from tactile_gym_servo_control.utils.image_transforms import process_image

from cri.robot import SyncRobot
from cri.controller import SimController as Controller

stimuli_path = os.path.join(os.path.dirname(__file__), 'stimuli')


def setup_embodiment(
    env_params={},
    sensor_params={},
    hover=[0, 0, -7.5, 0, 0, 0],
    tcp_pose=[0, 0, 0, 0, 0, 0],
    show_gui=True, 
    show_tactile=True,
    quick_mode=False
):
    env_params['stim_path'] = stimuli_path
    env_params['show_gui'] = show_gui
    env_params['show_tactile'] = show_tactile
    env_params['quick_mode'] = quick_mode

    # setup the robot
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
    embodiment.coord_frame = env_params['workframe']
    embodiment.tcp = tcp_pose

    embodiment.hover = np.array(hover)
    embodiment.show_gui = show_gui

    return embodiment


if __name__ == '__main__':
    pass
