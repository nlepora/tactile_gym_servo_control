
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
from cri.transforms import transform_pose, inv_transform_pose

from tactile_gym_servo_control.utils.setup_embodiment_sim import setup_embodiment
from tactile_gym_servo_control.learning.setup_learning import setup_task
from tactile_gym_servo_control.learning.setup_network import setup_network

from tactile_gym_servo_control.pushing.setup_pushing_sim import setup_tactile_pushing

from tactile_gym_servo_control.utils.controller import PIDController
from tactile_gym_servo_control.utils.utils_servoing import Slider, Model
from tactile_gym_servo_control.utils.plots_servoing import PlotContour3D as PlotContour

model_path = os.path.join(os.path.dirname(__file__), "../../example_models/sim/simple_cnn")
videos_path = os.path.join(os.path.dirname(__file__), "../../example_videos")


def run_push_control(
            embodiment, model, 
            ep_len=100,
            pid_pose_params={},
            ref_pose=[0, 0, 0, 0, 0, 0],
            pid_align_params={},
            ref_align=0,
            target_pose=[0, 0, 0, 0, 0, 0],
            record_vid=False
        ):
    
    rho_min = 60    # Target approach zone radius (mm) - turn off bearing controller
    rho_end = 20    # Tip radius 
    sensor_align_max = [10, 15, 5, 5, 5, 15]
    tap_move = [[10, 0, 0, 0, 0, 0], [5, 0, 0, 0, 0, 0]]

    # initialize peripherals
    if record_vid:
        render_frames = []
    # slider = Slider(ref_pose)
    # plotContour = PlotContour(embodiment.coord_frame)

    # initialise pose and move to initial pose
    pose = [0, 0, 0, 0, 0, 0]
    embodiment.move_linear(pose)

    # initialize controllers
    controller_pose =  PIDController(**pid_pose_params)
    controller_align = PIDController(**pid_align_params)

    # iterate through servo control
    for i in range(ep_len):

        # tapping motion move forward
        tcp_pose = embodiment.pose
        pose = inv_transform_pose(tap_move[0], tcp_pose) 
        embodiment.move_linear(pose)

        # get current tactile observation
        tactile_image = embodiment.sensor.process()

        # tapping motion move back
        pose = inv_transform_pose(tap_move[1], tcp_pose) 
        embodiment.move_linear(pose) 

        # predict pose from observation
        pred_pose = model.predict(tactile_image)
        depth, Ry, Rz = pred_pose[2], pred_pose[4], -pred_pose[3]
        pred_pose = np.array([depth, 0, 0, 0, Ry, Rz])

        # Compute servo control in sensor frame
        servo = controller_pose.update(pred_pose, ref_pose)

        # Compute servo control frame wrt work frame
        sensor_frame = embodiment.pose
        servo_frame = inv_transform_pose(servo, sensor_frame)
        servo_target = transform_pose(target_pose, servo_frame)

        # Compute target bearing (theta) and distance (rho) in servo control frame
        p_x, p_y = servo_target[:2]
        theta = np.degrees(np.arctan2(p_y, p_x))
        rho = np.sqrt(p_y**2 + p_x**2)

        # Disengage target tracking (PID controller 2) inside target approach zone
        if rho < rho_min:
            print("\n Disengaged target tracking...", end='')
            controller_align.kp, controller_align.ki, controller_align.kd = 0, 0, 0

        # Terminate push sequence when sensor tip is approximately on target
        if rho < rho_end:
            print("\n Target reached!")
            break
            
        # Compute target alignment control wrt servo control frame
        y_align = controller_align.update(theta, ref_align)  
        servo_align = [0, y_align, 0, 0, 0, 0]
        sensor_align = inv_transform_pose(servo_align, servo)
        work_align = inv_transform_pose(sensor_align, sensor_frame)

        # pause if any component of the target alignment control exceeds pre-defined limit
        if np.any(np.abs(sensor_align) > sensor_align_max):
            response = input("Alignment control exceeded. Enter 'q' to quit or RETURN to continue:")
            if response=='q':
                break

        # move
        embodiment.move_linear(work_align)

        # slider control
        # ref_pose = slider.read(ref_pose)
        
        # render frames
        if record_vid:
            render_img = embodiment.render()
            render_frames.append(render_img)

        # report
        print(f"\n Step: {i+1}, p_x: {p_x:.1f}, p_y: {p_y:.1f}, theta: {theta:.1f}, rho: {rho:.1f}")
        # plotContour.update(pose)
        # embodiment.controller._client._sim_env.arm.draw_TCP(lifetime=10.0)

    # finish
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
        help="Choose task from ['pushing_2d', 'pushing_3d'].",
        default=['pushing_3d']
    )
    parser.add_argument(
        '-s', '--stimuli',
        nargs='+',
        help="Choose stimulus from ['cube', 'circle'].",
        default=['cube']
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
    stimuli = args.stimuli
    device = args.device
    version = ''

    for task in tasks:

        # set saved model dir
        task += version
        model_dir = os.path.join(model_path, task)

        # load model and sensor params
        network_params = load_json_obj(os.path.join(model_dir, 'model_params'))
        learning_params = load_json_obj(os.path.join(model_dir, 'learning_params'))
        image_processing_params = load_json_obj(os.path.join(model_dir, 'image_processing_params'))
        sensor_params = load_json_obj(os.path.join(model_dir, 'sensor_params'))
        pose_params = load_json_obj(os.path.join(model_dir, 'pose_params'))
        out_dim, label_names = setup_task(task)

        # perform the servo control
        for stimulus in stimuli:

            env_params, control_params = setup_tactile_pushing[task](stimulus)

            embodiment = setup_embodiment(
                env_params,
                sensor_params=sensor_params, 
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

            run_push_control(
                embodiment,
                model,
                **control_params
            )
