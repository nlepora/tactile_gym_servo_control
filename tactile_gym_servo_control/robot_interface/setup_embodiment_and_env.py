import os

from tactile_gym_servo_control.robot_interface.setup_pybullet_env import setup_pybullet_env

stimuli_path = os.path.join(os.path.dirname(__file__), '../stimuli')


def setup_embodiment_and_env(
    stim_name="square",
    stim_pose=[600, 0, 12.5, 0, 0, 0],
    workframe=[600, 0, 52.5, -180, 0, 90],
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

    return embodiment


if __name__ == '__main__':
    pass
