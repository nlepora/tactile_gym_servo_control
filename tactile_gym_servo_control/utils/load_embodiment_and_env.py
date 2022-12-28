import os
import numpy as np

from tactile_gym_servo_control.utils.pybullet_utils import setup_pybullet_env

POSE_UNITS = np.array([1e-3, 1e-3, 1e-3, np.pi/180, np.pi/180, np.pi/180])

stimuli_path = os.path.join(os.path.dirname(__file__), '../stimuli')


def load_embodiment_and_env(
    stim_name="square", 
    show_gui=True, 
    show_tactile=True,
    quick_mode=False
):

    assert stim_name in ["square", "foil", "clover", "circle",
                         "saddle"], "Invalid Stimulus"

    tactip_params = {
        "name": "tactip",
        "type": "standard",
        "core": "no_core",
        "dynamics": {},
        "image_size": [128, 128],
        "turn_off_border": False,
    }

    # setup stimulus
    stimulus_pos = [600, 0, 12.5] * POSE_UNITS[:3]
    stimulus_rpy = [0, 0, 0] * POSE_UNITS[3:]
    stim_path = os.path.join(
        stimuli_path,
        stim_name,
        stim_name + ".urdf"
    )

    # set the work frame of the robot (relative to world frame)
    workframe_pos = [600, 0, 52.5] * POSE_UNITS[:3]
    workframe_rpy = [-180, 0, 90] * POSE_UNITS[3:]

    # setup robot data collection env
    embodiment, _ = setup_pybullet_env(
        stim_path,
        tactip_params,
        stimulus_pos,
        stimulus_rpy,
        workframe_pos,
        workframe_rpy,
        show_gui,
        show_tactile,
    )

    def move_linear(pose):
        pos = pose[:3]
        rpy = pose[3:]
        embodiment.move_linear(pos, rpy, quick_mode=quick_mode)

    embodiment.move = move_linear

    return embodiment


if __name__ == '__main__':
    pass
