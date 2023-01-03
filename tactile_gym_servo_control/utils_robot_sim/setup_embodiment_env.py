import os
import cv2

from tactile_gym_servo_control.utils_robot_sim.setup_pybullet_env import setup_pybullet_env

stimuli_path = os.path.join(os.path.dirname(__file__), 'stimuli')


def setup_embodiment_env(
    tactip_params = {},
    stim_name="square",
    stim_pose=[600, 0, 12.5, 0, 0, 0],
    workframe=[600, 0, 52.5, -180, 0, 90],
    hover=[0, 0, -7.5, 0, 0, 0],
    show_gui=True, 
    show_tactile=True,
    quick_mode=False
):

    assert stim_name in ["square", "foil", "clover", "circle",
                         "saddle"], "Invalid Stimulus"

    # setup stimulus in worldframe
    stim_path = os.path.join(
        stimuli_path, stim_name, stim_name + ".urdf"
    )

    # setup robot data collection env
    embodiment, _ = setup_pybullet_env(
        stim_path,
        tactip_params,
        stim_pose,
        workframe,
        show_gui,
        show_tactile,
        quick_mode
    )

    def sensor_process(outfile=None):
        img = embodiment.get_tactile_observation()
        if outfile is not None:
            cv2.imwrite(outfile, img)
        return img

    embodiment.sensor_process = sensor_process

    embodiment.hover = hover

    return embodiment


if __name__ == '__main__':
    pass
