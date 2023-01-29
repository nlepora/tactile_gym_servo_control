import cv2
import numpy as np

from tactile_gym_servo_control.utils.image_transforms import process_image
from tactile_gym_servo_control.utils.setup_pybullet_env import setup_pybullet_env

from cri.robot import SyncRobot
from cri.controller import Mg400Controller as Controller
# from cri.controller import DummyController as Controller

class Sensor:
    def __init__(self, 
        source=0, 
        exposure=-7, 
        **kwargs
    ):  
        self.cam = cv2.VideoCapture(source)
        self.cam.set(cv2.CAP_PROP_EXPOSURE, exposure)
        for _ in range(5): self.cam.read() # Hack - camera transient

    def read_image(self):
        self.cam.read() # Hack - throw one away - buffering issue
        _, img = self.cam.read()
        return img


def setup_embodiment(
    sensor_params={},
    workframe=[288, 0, -100, 0, 0, -90],
    linear_speed=10, 
    angular_speed=10,
    tcp_pose=[0, 0, 0, 0, 0, 0],
    hover=[0, 0, 7.5, 0, 0, 0], # positive for dobot
):
    # setup the tactile sensor
    embodiment = SyncRobot(Controller())
    sensor = Sensor(**sensor_params)

    def sensor_process(outfile=None):
        img = sensor.read_image()
        img = process_image(img, **sensor_params)
        if outfile is not None:
            cv2.imwrite(outfile, img)
        return img

    # setup the robot
    embodiment.sensor_process = sensor_process

    embodiment.coord_frame = workframe
    embodiment.tcp = tcp_pose
    
    embodiment.linear_speed = linear_speed
    embodiment.angular_speed = angular_speed
    
    embodiment.hover = np.array(hover)

    return embodiment


if __name__ == '__main__':
    pass
