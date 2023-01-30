import os
import numpy as np

from cri.robot import SyncRobot
from cri.controller import SimController as Controller
from tactile_gym_servo_control.utils.sensors import Sensor_sim as Sensor

stimuli_path = os.path.join(os.path.dirname(__file__), 'stimuli')


def setup_embodiment(
    env_params={},
    sensor_params={},
    hover=[0, 0, -7.5, 0, 0, 0]
):
    work_frame = env_params.get('work_frame', [600, 0, 0, 0, 0, 0])
    tcp_pose = env_params.get('tcp_pose', [0, 0, 0, 0, 0, 0])
    env_params['stim_path'] = stimuli_path

    # setup the embodiment
    embodiment = SyncRobot(Controller(sensor_params, env_params))   
    embodiment.sensor = Sensor(embodiment, sensor_params)

    # settings
    embodiment.coord_frame = work_frame
    embodiment.tcp = tcp_pose
    embodiment.hover = np.array(hover)

    return embodiment


if __name__ == '__main__':
    pass
