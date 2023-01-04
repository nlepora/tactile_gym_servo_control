import cv2
import numpy as np

from cri.robot import SyncRobot
from cri.controller import Mg400Controller as Controller
# from cri.controller import DummyController as Controller


class Sensor:
    def __init__(self,
        size=[128, 128], 
        crop=None, 
        exposure=-7, 
        source=0, 
        threshold=None
    ):  
        self.size, self.crop, self.threshold = size, crop, threshold
        self.cam = cv2.VideoCapture(source)
        self.cam.set(cv2.CAP_PROP_EXPOSURE, exposure)
        for _ in range(5): self.cam.read() # Hack - camera transient 
    def process(self):
        _, img = self.cam.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.crop is not None:
            x0, y0, x1, y1 = self.crop
            img = img[y0:y1, x0:x1]
        if self.threshold is not None:
            img = cv2.medianBlur(img, 5)
            img = cv2.adaptiveThreshold(img, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 
                *self.threshold
            )
        if self.size is not None:
            img = cv2.resize(img, 
                self.size, interpolation=cv2.INTER_AREA
            )
        return img


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

    embodiment.coord_frame = workframe

    # setup the tactip
    sensor = Sensor(**sensor_params)

    def sensor_process(outfile=None):
        img = sensor.process()
        if outfile is not None:
            cv2.imwrite(outfile, img)
        return img

    embodiment.sensor_process = sensor_process

    embodiment.hover = np.array(hover)

    return embodiment


if __name__ == '__main__':
    pass
