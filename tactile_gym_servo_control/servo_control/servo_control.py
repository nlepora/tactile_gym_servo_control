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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tactile_gym.utils.general_utils import load_json_obj

from tactile_gym_servo_control.data_collection.data_collection_utils import load_embodiment_and_env

from tactile_gym_servo_control.learning.learning_utils import import_task
from tactile_gym_servo_control.learning.networks import create_model

from tactile_gym_servo_control.servo_control.servo_control_utils import add_gui
from tactile_gym_servo_control.servo_control.servo_control_utils import get_prediction
from tactile_gym_servo_control.servo_control.servo_control_utils import compute_target_pose

from tactile_gym_servo_control.servo_control.setup_servo_control import setup_surface_3d_servo_control
from tactile_gym_servo_control.servo_control.setup_servo_control import setup_edge_2d_servo_control
from tactile_gym_servo_control.servo_control.setup_servo_control import setup_edge_3d_servo_control
from tactile_gym_servo_control.servo_control.setup_servo_control import setup_edge_5d_servo_control

model_path = os.path.join(os.path.dirname(__file__), "../../example_models/nature_cnn")
videos_path = os.path.join(os.path.dirname(__file__), "../../example_videos")


def run_servo_control(
            embodiment,
            trained_model,
            image_processing_params,
            p_gains=np.zeros(6),
            label_names=[],
            pose_limits=[],
            ref_pose=np.zeros(6),
            ep_len=400,
            init_pose=np.zeros(6),
            record_vid=False,
            show_gui=True
        ):

    if show_gui:
        ref_pose_ids = add_gui(embodiment, ref_pose)
        
    if record_vid:
        render_frames = []

    # move to initial pose
    embodiment.move(init_pose)

    # iterate through servo control
    for i in range(ep_len):

        # get current tactile observation
        tactile_image = embodiment.get_tactile_observation()

        # get current TCP pose
        tcp_pose = embodiment.get_tcp_pose()

        # predict pose from observation
        pred_pose = get_prediction(
            trained_model,
            tactile_image,
            image_processing_params,
            label_names,
            pose_limits,
            device
        )

        # compute new pose to move to
        target_pose = compute_target_pose(
            pred_pose, ref_pose, p_gains, tcp_pose
        )

        # move to new pose
        embodiment.move(target_pose)

        # draw TCP frame
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
        default='cpu'
    )

    # parse arguments
    args = parser.parse_args()
    tasks = args.tasks
    device = args.device

    setup_servo_control = {
        "surface_3d": setup_surface_3d_servo_control,
        "edge_2d": setup_edge_2d_servo_control,
        "edge_3d": setup_edge_3d_servo_control,
        "edge_5d": setup_edge_5d_servo_control
    }

    for task in tasks:

        # set save dir
        save_dir_name = os.path.join(model_path, task)

        # load params
        model_params = load_json_obj(os.path.join(save_dir_name, 'model_params'))
        learning_params = load_json_obj(os.path.join(save_dir_name, 'learning_params'))
        image_processing_params = load_json_obj(os.path.join(save_dir_name, 'image_processing_params'))
        pose_limits_dict = load_json_obj(os.path.join(save_dir_name, 'pose_limits'))
 
        # get limits and labels used during training
        out_dim, label_names = import_task(task)
        pose_limits = [pose_limits_dict['pose_llims'], pose_limits_dict['pose_ulims']]

        # setup the task
        stim_names, ep_len, init_poses, ref_pose, p_gains = setup_servo_control[task]()

        # perform the servo control
        for j, stim_name in enumerate(stim_names):

            embodiment = load_embodiment_and_env(stim_name)
            init_pose = init_poses[j]
            
            trained_model = create_model(
                image_processing_params['dims'],
                out_dim,
                model_params,
                saved_model_dir=save_dir_name,
                device=device
            )
            trained_model.eval()

            run_servo_control(
                embodiment,
                trained_model,
                image_processing_params,
                p_gains=p_gains,
                label_names=label_names,
                pose_limits=pose_limits,
                ref_pose=ref_pose,
                ep_len=ep_len,
                init_pose=init_pose,
                record_vid=True
            )
