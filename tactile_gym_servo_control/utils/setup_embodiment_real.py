import cv2
import numpy as np

from cri.robot import SyncRobot
from cri.controller import Mg400Controller as Controller
from tactile_gym_servo_control.utils.image_transforms import process_image


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
    env_params={},    
    hover=[0, 0, 7.5, 0, 0, 0], # positive for dobot
):
    workframe = env_params.get('workframe', [288, 0, -100, 0, 0, -90])
    tcp_pose = env_params.get('tcp_pose', [0, 0, 0, 0, 0, 0])
    linear_speed = env_params.get('linear_speed', 10)
    angular_speed = env_params.get('angular_speed', 10)

    # setup the embodiment
    embodiment = SyncRobot(Controller())
    sensor = Sensor(**sensor_params)

    # setup the tactile sensor
    def sensor_process(outfile=None):
        img = sensor.read_image()
        img = process_image(img, **sensor_params)
        if outfile is not None:
            cv2.imwrite(outfile, img)
        return img

    embodiment.sensor_process = sensor_process

    # settings
    embodiment.coord_frame = workframe
    embodiment.tcp = tcp_pose
    embodiment.linear_speed = linear_speed
    embodiment.angular_speed = angular_speed
    embodiment.hover = np.array(hover)

    return embodiment


if __name__ == '__main__':
    pass
