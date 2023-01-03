import numpy as np

from cri.robot import SyncRobot
from cri.controller import Mg400Controller as Controller
# from cri.controller import DummyController as Controller
from vsp.video_stream import CvImageOutputFileSeq, CvVideoDisplay, CvPreprocVideoCamera   
from vsp.processor import CameraStreamProcessor, AsyncProcessor


def make_sensor(
    size=[128, 128], 
    crop=None, 
    exposure=-7, 
    source=0, 
    threshold=None
):  
    camera = CvPreprocVideoCamera(
        size, crop, threshold, exposure=exposure, source=source
    )
    
    for _ in range(5): camera.read() # Hack - camera transient   
    
    return AsyncProcessor(CameraStreamProcessor(
            camera=camera,
            display=CvVideoDisplay(name='sensor'),
            writer=CvImageOutputFileSeq())
    )


def setup_embodiment_env(
    tactip_params={},
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
    sensor = make_sensor(**tactip_params)

    def sensor_process(outfile=None):
        img = sensor.process(
            num_frames=1, start_frame=1, outfile=outfile
        )
        return img[0,:,:,0]

    embodiment.sensor_process = sensor_process

    embodiment.hover = hover

    return embodiment


if __name__ == '__main__':
    pass
