"""
python servo_control.py -t surface_3d
python servo_control.py -t edge_2d
python servo_control.py -t edge_3d
python servo_control.py -t edge_5d
python servo_control.py -t surface_3d edge_2d edge_3d edge_5d
"""

import os
import argparse
import numpy as np
import imageio

from tactile_gym.utils.general_utils import load_json_obj

from tactile_gym_servo_control.utils.setup_embodiment_sim import setup_embodiment
from tactile_gym_servo_control.learning.setup_learning import setup_task
from tactile_gym_servo_control.learning.setup_network import setup_network

from tactile_gym_servo_control.servo_control.setup_servo_control_sim import setup_servo_control
# from tactile_gym_servo_control.servo_control.utils_servo_control import Slider
from tactile_gym_servo_control.servo_control.utils_servo_control import Model
from tactile_gym_servo_control.servo_control.utils_plots import PlotContour3D as PlotContour
from tactile_gym_servo_control.utils.pose_transforms import transform_pose, inv_transform_pose

np.set_printoptions(precision=1, suppress=True)

model_path = os.path.join(os.path.dirname(__file__), "../../example_models/sim/simple_cnn")
videos_path = os.path.join(os.path.dirname(__file__), "../../example_videos")

model_version = ''


def run_servo_control(
            embodiment, model,
            ep_len=100,
            ref_pose=[0, 0, 0, 0, 0, 0],
            p_gains=[0, 0, 0, 0, 0, 0],
            i_gains=[0, 0, 0, 0, 0, 0],
            i_clip=[-np.inf, np.inf],
            record_vid=False
        ):

    if record_vid:
        render_frames = []

    # if embodiment.show_gui:
    #     slider = Slider(embodiment.slider, ref_pose)

    plotContour = PlotContour(embodiment.workframe)

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
        tactile_image = embodiment.sensor_process()

        # predict pose from observation
        pred_pose = model.predict(tactile_image)

        # find deviation of prediction from reference
        delta = transform_pose(ref_pose, pred_pose)

        # apply pi(d) control to reduce delta
        int_delta += delta
        int_delta = np.clip(int_delta, *i_clip)

        output = p_gains * delta  +  i_gains * int_delta 
        
        # new pose combines output pose with tcp_pose 
        tcp_pose = embodiment.pose # need to test on real
        pose = inv_transform_pose(output, tcp_pose)

        # move to new pose
        embodiment.move_linear(pose)

        # slider control
        # if embodiment.show_gui:
        #     ref_pose = slider.slide(ref_pose)
           
        # show tcp if sim
        # if embodiment.show_gui and embodiment.sim:
        #     embodiment.controller._client._sim_env.arm.draw_TCP(lifetime=10.0)
        
        # render frames
        if record_vid:
            render_img = embodiment.render()
            render_frames.append(render_img)

        # report
        print(f'\nstep {i+1}: pose: {pose}', end='')
        plotContour.update(pose)

    # move to above final pose
    embodiment.move_linear(pose + hover)
    embodiment.close()

    if record_vid:
        imageio.mimwrite(
            os.path.join(videos_path, "render.mp4"),
            np.stack(render_frames),
            fps=24
        )


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
        default=['bowl']
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

        # set saved model dir
        model_dir = os.path.join(model_path, task + model_version)

        # load model and sensor params
        network_params = load_json_obj(os.path.join(model_dir, 'model_params'))
        learning_params = load_json_obj(os.path.join(model_dir, 'learning_params'))
        image_processing_params = load_json_obj(os.path.join(model_dir, 'image_processing_params'))
        sensor_params = load_json_obj(os.path.join(model_dir, 'sensor_params'))
        pose_params = load_json_obj(os.path.join(model_dir, 'pose_params'))
        out_dim, label_names = setup_task(task)

        # perform the servo control
        for stimulus in stimuli:

            env_params, control_params = setup_servo_control[task](stimulus)

            embodiment = setup_embodiment(
                env_params, 
                sensor_params, 
                quick_mode=True
            )

            network = setup_network(
                image_processing_params['dims'],
                out_dim,
                network_params,
                saved_model_dir=model_dir,
                device=device
            )
            network.eval()

            model = Model(
                network,
                image_processing_params,
                pose_params,
                label_names,
                device=device
            )

            run_servo_control(
                embodiment,
                model,
                **control_params
            )
