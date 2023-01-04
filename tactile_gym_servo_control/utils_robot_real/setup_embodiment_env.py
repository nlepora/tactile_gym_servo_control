import cv2
import numpy as np

from tactile_gym_servo_control.utils.image_transforms import Sensor, process_image

from cri.robot import SyncRobot
from cri.controller import Mg400Controller as Controller
# from cri.controller import DummyController as Controller


def setup_embodiment_env(
    sensor_params={},
    workframe=[288, 0, -100, 0, 0, -90],
    linear_speed=10, 
    angular_speed=10,
    tcp_pose=[0, 0, 0, 0, 0, 0],
    hover=[0, 0, 7.5, 0, 0, 0] # positive for dobot
):

    # setup the robot
    embodiment = SyncRobot(Controller())

    embodiment.coord_frame = workframe
    embodiment.linear_speed = linear_speed
    embodiment.angular_speed = angular_speed
    embodiment.tcp = tcp_pose
    embodiment.hover = np.array(hover)
    embodiment.name = 'real'

    # setup the tactip
    sensor = Sensor(**sensor_params)

    def sensor_process(outfile=None):
        img = sensor.load()
        img = process_image(img, **sensor_params)
        if outfile is not None:
            cv2.imwrite(outfile, img)
        return img

    embodiment.sensor_process = sensor_process

    return embodiment


if __name__ == '__main__':
    pass
