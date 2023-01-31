"""
python servo_control_manual.py -t surface_3d
python servo_control_manual.py -t edge_2d
python servo_control_manual.py -t edge_3d
python servo_control_manual.py -t edge_5d
python servo_control_manual.py -t surface_3d edge_2d edge_3d edge_5d
"""

import argparse
import numpy as np

from cri.transforms import inv_transform_pose

from tactile_gym_servo_control.utils.setup_embodiment_sim import setup_embodiment
from tactile_gym_servo_control.servo_control.setup_servo_control_sim import setup_servo_control
from tactile_gym_servo_control.utils.utils_servo_control import ManualControl
from tactile_gym_servo_control.utils.plots_servo_control import PlotContour3D as PlotContour

np.set_printoptions(precision=1, suppress=True)


def run_manual_servo_control(
            embodiment,
            ep_len=10000,
            ref_pose=None,
            p_gains=[0, 0, 0, 0, 0, 0],
            i_gains=[0, 0, 0, 0, 0, 0],
            i_clip=[-np.inf, np.inf],
            i_leak=0.9
        ):

    # initialize slider and plot
    manual = ManualControl(embodiment)
    plotContour = PlotContour(embodiment.coord_frame)

    # initialise pose and integral term
    pose = [0, 0, 0, 0, 0, 0]
    int_delta = [0, 0, 0, 0, 0, 0]

    # move to initial pose from above workframe
    hover = embodiment.hover
    embodiment.move_linear(pose + hover)
    embodiment.move_linear(pose)

    # iterate through servo control
    for i in range(ep_len):

        # get current tactile observation
        tactile_image = embodiment.sensor.process()

        # manually control robot
        delta = manual.spacemouse()
        if delta is None: 
            break # quit

        # apply pi(d) control to reduce delta
        int_delta = i_leak * np.array(int_delta) + delta
        int_delta = np.clip(int_delta, *i_clip)

        output = p_gains * delta  +  i_gains * int_delta 
        
        # new pose combines output pose with tcp_pose 
        tcp_pose = embodiment.pose # need to test on real
        pose = inv_transform_pose(output, tcp_pose)

        # move to new pose
        embodiment.move_linear(pose)

        # report
        print(f'\nstep {i+1}: pose: {pose}', end='')
        plotContour.update(pose)

    # move to above final pose
    embodiment.move_linear(pose + hover)
    embodiment.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--tasks',
        nargs='+',
        help="Choose task from ['surface_3d', 'edge_2d', 'edge_3d', 'edge_5d'].",
        default=['surface_3d']
    )
    parser.add_argument(
        '-s', '--stimuli',
        nargs='+',
        help="Choose stimulus from ['circle', 'square', 'clover', 'foil', 'saddle', 'bowl'].",
        default=['saddle']
    )
    parser.add_argument(
        '-d', '--device',
        type=str,
        help="Choose device from ['cpu', 'cuda'].",
        default='cuda'
    )

    # parse arguments
    args = parser.parse_args()
    tasks = args.tasks
    stimuli = args.stimuli
    device = args.device

    for task in tasks:
        for stimulus in stimuli:

            env_params, control_params = setup_servo_control[task](stimulus)
            
            env_params.update({
                'show_gui': True, 'show_tactile': True, 'quick_mode': False
            })
            
            embodiment = setup_embodiment(
                env_params
            )

            run_manual_servo_control(
                embodiment,
                **control_params
            )
