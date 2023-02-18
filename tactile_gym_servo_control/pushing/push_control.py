
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
from tactile_gym_servo_control.pushing.controller import PIDController
from tactile_gym_servo_control.pushing.utils_pushing import Namespace, transform_euler, inv_transform_euler

from tactile_gym_servo_control.utils.utils_servoing import Slider, Model
from tactile_gym_servo_control.utils.plots_servoing import PlotContour3D as PlotContour


np.set_printoptions(precision=1, suppress=True)

model_path = os.path.join(os.path.dirname(__file__), "../../example_models/sim/simple_cnn")
videos_path = os.path.join(os.path.dirname(__file__), "../../example_videos")

# Helper function/closure for computing SE(3) error
def se3_error(y, r):
    y_inv = transform_euler([0, 0, 0, 0, 0, 0], y, 'sxyz')
    err = inv_transform_euler(r, y_inv, 'sxyz')
    return err

def run_tactile_pushing(
            embodiment, model,
            ep_len=100,
            ref_pose=[0, 0, 0, 0, 0, 0],
            p_gains=[0, 0, 0, 0, 0, 0],
            i_gains=[0, 0, 0, 0, 0, 0],
            i_clip=[-np.inf, np.inf],
            work_target_pose=[],
            record_vid=False
        ):
    
    robot_axes = 'sxyz'

    sensor_tap_move = [[10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [5.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    kp1 = [0.9, 0.0, 0.0, 0.0, 0.0, 0.9]                  # irregular objects / flat surface
    # ki1 = [0.0, 0.0, 0.1, 0.1, 0.1, 0.0]                # general/curved surfaces
    ki1 = [0.1, 0.0, 0.0, 0.0, 0.0, 0.1]                  # irregular objects / flat surface
    ei_min1 = [-5.0, -5.0, -5.0, -25.0, -25.0, -25.0]     # integral error clipping
    ei_max1 = [5.0, 5.0, 5.0, 25.0, 25.0, 25.0]           # integral error clipping
    alpha1 = 0.5                                          # differential error filtering coeff
    
    ref_sensor_pose = [1.5, 0.0, 0.0, 0.0, 1.0, 0.0]      # flat surface ref pose
    # m.ref_sensor_pose = [0.0, 0.0, 3.0, 0.0, 3.0, 0.0]    # convex surface ref pose
    # m.ref_sensor_pose = [0.0, 0.0, 3.0, 0.0, 0.0, 0.0]    # concave surface ref pose

    # Target approach zone radius (mm)
    rho_min = 60.0

    sensor_tip_radius = 20.0

    sensor_align_max_abs = [10.0, 15.0, 5.0, 5.0, 5.0, 15.0]

    # tcp_offset = [0.0, 0.0, 0.0, 90.0, 0.0, 180.0]

    # PID controller #2 params
    kp2 = 0.2             # proportional gain
    kd2 = 0.9             # differential gain
    ep_min2 = -10.0       # proportional error clipping
    ep_max2 = 10.0        # proportional error clipping
    alpha2 = 0.5          # differential error filtering coeff
    ref_theta = 0.0       # target bearing (in servo frame)

    pid1 = PIDController(kp=kp1, ki=ki1, ei_min=ei_min1, ei_max=ei_max1,
                         alpha=alpha1, error=se3_error)
    pid2 = PIDController(kp=kp2, kd=kd2, ep_min=ep_min2, ep_max=ep_max2,
                         alpha=alpha2)

    if record_vid:
        render_frames = []

    if embodiment.show_gui:
        slider = Slider(embodiment.slider, ref_pose)

    # plotContour = PlotContour(embodiment.workframe)#, embodiment.stim_name)

    # initialise pose and integral term
    pose = [0, 0, 0, 0, 0, 0]
    int_delta = [0, 0, 0, 0, 0, 0]

    # move to initial pose from above workframe
    hover = embodiment.hover
    embodiment.move_linear(pose + hover)

    # joint_pos, _ = embodiment.arm.get_current_joint_pos_vel()
    # print(joint_pos)

    embodiment.move_linear(pose)

    # while True:
    #     embodiment.move_linear(pose)

    def get_tcp_pose():
        if embodiment.sim:
            return embodiment.get_tcp_pose()
        else:
            return pose


    # iterate through servo control
    for i in range(ep_len):

        # Option 1: Calculate output pose with tcp_pose to move forward 
        # tcp_pose = get_tcp_pose()
        # output = [5, 5, 0, 0, 0, 0] # pose to move to in sensor frame
        # pose = inv_transform_pose(output[0], tcp_pose) # pose to move to in workframe
        # embodiment.move_linear(pose) # move to pose in workframe

        # Option 2: Update workframe as current tcp pose and move forward by defining pose in sensor frame
        # tcp_pose = get_tcp_pose()
        # embodiment.set_workframe(embodiment.get_tcp_pose_worldframe()) # update workframe as current tcp pose
        # pose = [5, 0, 0, 0, 0, 0] # pose to move to in sensor frame
        # embodiment.move_linear(pose) # move to pose in workframe (sensor frame)

        # Tapping motion move forward
        tcp_pose = get_tcp_pose()
        # pose_sensor = transform_pose(sensor_tap_move[0], tcp_offset)
        pose = inv_transform_pose(sensor_tap_move[0], tcp_pose.copy()) 
        embodiment.move_linear(pose) # move to pose in workframe

        # get current tactile observation
        tactile_image = embodiment.sensor_process()

        # Tapping motion move back
        # pose_sensor = transform_pose(sensor_tap_move[1], tcp_offset)
        pose = inv_transform_pose(sensor_tap_move[1], tcp_pose.copy()) 
        embodiment.move_linear(pose) # move to pose in workframe


        # predict pose from observation
        pred_pose = model.predict(tactile_image)
        # print('z: {}, Rx: {}, Ry: {}'.format(pred_pose[2], pred_pose[3], pred_pose[4]))
        depth, Ry, Rz = pred_pose[2], pred_pose[4], -pred_pose[3]
        pred_pose = np.array([depth, 0, 0, 0, Ry, Rz])

        # Compute servo control in sensor frame
        sensor_servo = pid1.update(y=pred_pose, r=ref_sensor_pose)
        # print('sensor_servo: {}'.format(sensor_servo))

        # Compute servo control frame wrt work frame
        work_sensor_frame = get_tcp_pose()
        work_servo_frame = inv_transform_euler(sensor_servo, work_sensor_frame.copy(), robot_axes)
        servo_target = transform_euler(work_target_pose, work_servo_frame.copy(), robot_axes)

        # Compute target bearing (theta) and distance (rho) in servo control frame
        p_x, p_y = servo_target[0:2]
        theta = np.degrees(np.arctan2(p_y, p_x))
        rho = np.sqrt(p_y ** 2 + p_x ** 2)

        print((f"Step: {i}, p_x: {float(p_x):.2f}, p_y: {float(p_y):.2f} "
        f"theta: {float(theta):.2f}, rho: {float(rho):.2f}"))

        # Disengage target tracking (PID controller 2) inside target approach zone
        if rho < rho_min:
            pid2.kp = 0.0
            pid2.ki = 0.0
            pid2.kd = 0.0

        # Compute target alignment control wrt servo control frame
        servo_align = np.zeros(6)
        servo_align[1] = pid2.update(y=theta, r=ref_theta)  # correction to align sensor with target
        sensor_align = inv_transform_euler(servo_align, sensor_servo.copy(), robot_axes)

        work_align = inv_transform_euler(sensor_align, work_sensor_frame.copy(), robot_axes)

        # Terminate push sequence when sensor tip is approximately on target
        if rho <= sensor_tip_radius:
            print("Target reached!")
            break

        # Pause if any component of the target alignment control exceeds pre-defined limit
        if np.any(np.abs(sensor_align) > sensor_align_max_abs):
            with np.printoptions(precision=2, suppress=True):
                print(f"Alignment control has exceeded max abs value: {sensor_align}")
            response = input("Enter 'q' to quit or RETURN to continue: ")
            if response == 'q':
                break

        embodiment.move_linear(work_align)

        # slider control
        if embodiment.show_gui:
            ref_pose = slider.slide(ref_pose)
           
        # show tcp if sim
        if embodiment.show_gui and embodiment.sim:
            embodiment.arm.draw_TCP(lifetime=10.0)
        
        # render frames
        if record_vid:
            render_img = embodiment.render()
            render_frames.append(render_img)

        # report
        print(f'\nstep {i+1}: pose: {pose}', end='')
        # plotContour.update(pose)

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
        help="Choose task from ['pushing_2d', 'pushing_3d'].",
        default=['pushing_3d']
    )
    parser.add_argument(
        '-s', '--stimuli',
        nargs='+',
        help="Choose stimulus from ['cube'].",
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

            cam_params = {
                'image_size': [512, 512],
                'dist': 1.2,
                'yaw': 90.0,
                'pitch': -45.0,
                'pos': [0.1, 0.0, -0.35],
                'fov': 75.0,
                'near_val': 0.1,
                'far_val': 100.0
            }

            env_params.update({
                'cam_params': cam_params,
                'quick_mode': False
            })

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

            run_tactile_pushing(
                embodiment,
                model,
                **control_params
            )
