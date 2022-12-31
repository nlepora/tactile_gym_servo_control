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

from tactile_gym_servo_control.utils_robot_sim.setup_embodiment_env import setup_embodiment_env
from tactile_gym_servo_control.learning.setup_learning import setup_task
from tactile_gym_servo_control.learning.setup_network import setup_network

from tactile_gym_servo_control.servo_control.setup_servo_control import setup_servo_control
from tactile_gym_servo_control.servo_control.utils_servo_control import add_gui
from tactile_gym_servo_control.servo_control.utils_servo_control import Model
from tactile_gym_servo_control.utils_robot_sim.robot_embodiment import transform, inv_transform

data_path = os.path.join(os.path.dirname(__file__), "../../example_data/sim")
model_path = os.path.join(os.path.dirname(__file__), "../../example_models/sim/simple_cnn")
videos_path = os.path.join(os.path.dirname(__file__), "../../example_videos")


def run_servo_control(
            embodiment, model,
            ep_len=100,
            ref_pose=[0, 0, 0, 0, 0, 0],
            p_gains=[0, 0, 0, 0, 0, 0],
            i_gains=[0, 0, 0, 0, 0, 0],
            i_clip=[-np.inf, np.inf],
            init_pose=[0, 0, 0, 0, 0, 0],
            record_vid=False,
            show_gui=True
        ):

    if show_gui:
        ref_pose_ids = add_gui(embodiment, ref_pose)
        
    if record_vid:
        render_frames = []

    # move to initial pose
    embodiment.move_linear(init_pose)

    # iterate through servo control
    int_delta = [0, 0, 0, 0, 0, 0]

    for _ in range(ep_len):

        # get current tactile observation
        tactile_image = embodiment.get_tactile_observation()

        # get current TCP pose
        tcp_pose = embodiment.get_tcp_pose()

        # predict pose from observation
        pred_pose = model.predict(tactile_image)

        # find deviation of prediction from reference
        delta = transform(ref_pose, pred_pose)

        # apply pi(d) control to reduce delta
        int_delta += delta
        int_delta = np.clip(int_delta, *i_clip)

        output = p_gains * delta  +  i_gains * int_delta 
        
        # new pose combines output pose with tcp_pose 
        pose = inv_transform(output, tcp_pose)

        # move to new pose
        embodiment.move_linear(pose)

        # display TCP frame
        embodiment.arm.draw_TCP(lifetime=10.0)

        # show gui
        if show_gui:
            for j in range(len(ref_pose)):
                ref_pose[j] = embodiment._pb.readUserDebugParameter(ref_pose_ids[j]) 

        # render frames
        if record_vid:
            render_img = embodiment.render()
            render_frames.append(render_img)

        # quit if press 'q' key
        keys = embodiment._pb.getKeyboardEvents()
        if ord('q') in keys and embodiment._pb.KEY_WAS_TRIGGERED:
            exit()

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
        '-d', '--device',
        type=str,
        help="Choose device from ['cpu', 'cuda'].",
        default='cuda'
    )

    # parse arguments
    args = parser.parse_args()
    tasks = args.tasks
    device = args.device

    for task in tasks:

        # set save dir
        save_dir = os.path.join(model_path, task)
        data_dir = os.path.join(data_path, task)

        # load params
        network_params = load_json_obj(os.path.join(save_dir, 'model_params'))
        learning_params = load_json_obj(os.path.join(save_dir, 'learning_params'))
        image_processing_params = load_json_obj(os.path.join(save_dir, 'image_processing_params'))
        pose_limits_dict = load_json_obj(os.path.join(save_dir, 'pose_limits'))
        tactip_params = load_json_obj(os.path.join(data_dir, 'train', 'tactip_params'))

        # get limits and labels used during training
        out_dim, label_names = setup_task(task)
        pose_limits = [pose_limits_dict['pose_llims'], pose_limits_dict['pose_ulims']]

        # setup servo control for the task
        stim_names, init_poses, ep_len, \
            ref_pose, p_gains, i_gains, i_clip = setup_servo_control[task]()

        # perform the servo control
        for init_pose, stim_name in zip(init_poses, stim_names):

            embodiment = setup_embodiment_env(
                tactip_params,
                stim_name,
                quick_mode=True
            )
            
            network = setup_network(
                image_processing_params['dims'],
                out_dim,
                network_params,
                saved_model_dir=save_dir,
                device=device
            )
            network.eval()

            model = Model(
                network,
                image_processing_params,
                label_names,
                pose_limits,
                device=device
            )

            run_servo_control(
                embodiment,
                model,
                ep_len,
                ref_pose=ref_pose,
                p_gains=p_gains,
                i_gains=i_gains,
                i_clip=i_clip,
                init_pose=init_pose,
                record_vid=False
            )
