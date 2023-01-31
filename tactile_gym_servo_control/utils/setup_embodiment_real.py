import numpy as np

from cri.robot import SyncRobot
from cri.controller import Mg400Controller as Controller
from tactile_gym_servo_control.utils.sensors import Sensor_real as Sensor


def setup_embodiment(
    env_params={},    
    sensor_params={},
    hover=[0, 0, 7.5, 0, 0, 0], # positive for dobot
):
    workframe = env_params.get('workframe', [288, 0, -100, 0, 0, -90])
    tcp_pose = env_params.get('tcp_pose', [0, 0, 0, 0, 0, 0])
    linear_speed = env_params.get('linear_speed', 10)
    angular_speed = env_params.get('angular_speed', 10)

    # setup the embodiment
    embodiment = SyncRobot(Controller())
    embodiment.sensor = Sensor(sensor_params)

    # settings
    embodiment.coord_frame = workframe
    embodiment.tcp = tcp_pose
    embodiment.linear_speed = linear_speed
    embodiment.angular_speed = angular_speed
    embodiment.hover = np.array(hover)

    return embodiment


if __name__ == '__main__':
    pass
