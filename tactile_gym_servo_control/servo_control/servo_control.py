import os
import numpy as np
import torch
from torch.autograd import Variable
from pytorch_model_summary import summary
import imageio


from tactile_gym.utils.general_utils import load_json_obj

from tactile_gym_servo_control.learning.learning_utils import decode_pose
from tactile_gym_servo_control.learning.learning_utils import POSE_LABEL_NAMES, POS_LABEL_NAMES, ROT_LABEL_NAMES
from tactile_gym_servo_control.cri_wrapper.cri_robot_arm import quat2euler, euler2quat, transform, inv_transform
from tactile_gym_servo_control.utils.pybullet_utils import setup_pybullet_env
from tactile_gym_servo_control.learning.networks import CNN
from tactile_gym_servo_control.utils.image_transforms import process_image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_robot_and_env(stim_name="square"):

    assert stim_name in ["square", "foil",
                         "clover", "circle",
                         "saddle"], "Invalid Stimulus"

    tactip_params = {
        "name": "tactip",
        "type": "standard",
        "core": "no_core",
        "dynamics": {},
        "image_size": [128, 128],
        "turn_off_border": False,
    }

    show_gui = True
    show_tactile = True

    # setup stimulus
    stimulus_pos = [0.6, 0.0, 0.0125]
    stimulus_rpy = [0, 0, 0]
    stim_path = os.path.join(
        os.path.dirname(__file__),
        "../stimuli/",
        stim_name,
        stim_name + ".urdf"
    )

    # set the work frame of the robot (relative to world frame)
    workframe_pos = [0.6, 0.0, 0.0525]
    workframe_rpy = [-np.pi, 0.0, np.pi / 2]

    # setup robot data collection env
    robot, _ = setup_pybullet_env(
        stim_path,
        tactip_params,
        stimulus_pos,
        stimulus_rpy,
        workframe_pos,
        workframe_rpy,
        show_gui,
        show_tactile,
    )

    return robot


def load_nn_model(save_dir_name, out_dim, device='cpu'):

    # load params
    learning_params = load_json_obj(
        os.path.join(save_dir_name, 'learning_params')
    )
    image_processing_params = load_json_obj(
        os.path.join(save_dir_name, 'image_processing_params')
    )

    # create and load model
    model = CNN(
        out_dim, image_processing_params['dims'], learning_params
    ).to(device)

    model.load_state_dict(torch.load(os.path.join(
        save_dir_name, 'best_model.pth'), map_location='cpu')
    )

    model.eval()

    print(summary(
        model,
        torch.zeros((1, 1, *image_processing_params['dims'])).to(device),
        show_input=True
    ))

    return model, learning_params, image_processing_params


def get_prediction(
    trained_model,
    tactile_image,
    image_processing_params,
    label_names,
    pose_limits,
    device='cpu'
):

    # process image (as done for training)
    processed_image = process_image(
        tactile_image,
        gray=False,
        bbox=image_processing_params['bbox'],
        dims=image_processing_params['dims'],
        stdiz=image_processing_params['stdiz'],
        normlz=image_processing_params['normlz'],
        thresh=image_processing_params['thresh'],
    )

    # channel first for pytorch
    processed_image = np.rollaxis(processed_image, 2, 0)

    # add batch dim
    processed_image = processed_image[np.newaxis, ...]

    # perform inference with the trained model
    model_input = Variable(torch.from_numpy(processed_image)).float().to(device)
    raw_predictions = trained_model(model_input)

    # decode the prediction
    predictions_dict = decode_pose(raw_predictions, label_names, pose_limits)

    print("")
    print("Predictions")
    predictions_arr = np.zeros(6)
    for label_name in label_names:
        if label_name in POS_LABEL_NAMES:
            predicted_val = predictions_dict[label_name].detach().cpu().numpy() * 0.001
        if label_name in ROT_LABEL_NAMES:
            predicted_val = predictions_dict[label_name].detach().cpu().numpy() * np.pi / 180

        print(label_name, predicted_val)
        predictions_arr[POSE_LABEL_NAMES.index(label_name)] = predicted_val

    return predictions_arr


def compute_target_pose(pred_pose, ref_pose, p_gains, tcp_pose):
    """
    Compute workframe pose for maintaining reference pose from predicted pose
    """

    # calculate deltas between reference and predicted pose
    ref_pose_q = euler2quat(ref_pose)
    pred_pose_q = euler2quat(pred_pose)
    pose_deltas = quat2euler(transform(ref_pose_q, pred_pose_q))

    # control signal in tcp frame (e.g frame of training data labels)
    control_signal = p_gains * pose_deltas

    # transform control signal from feature frame to TCP frame
    control_signal_q = euler2quat(control_signal)
    tcp_pose_q = euler2quat(tcp_pose)
    target_pose = quat2euler(inv_transform(control_signal_q, tcp_pose_q))

    return target_pose


def run_servo_control(
            robot,
            trained_model,
            image_processing_params,
            ref_pose=np.zeros(6),
            p_gains=np.zeros(6),
            label_names=[],
            pose_limits=[],
            ep_len=400,
            quick_mode=True,
            record_vid=False,
        ):

    if record_vid:
        render_frames = []

    for i in range(ep_len):

        # get current tactile observation
        tactile_image = robot.get_tactile_observation()

        # get current TCP pose
        tcp_pose = robot.get_tcp_pose()

        # predict pose from observation
        pred_pose = get_prediction(
            trained_model, tactile_image, image_processing_params, label_names, pose_limits
        )

        # compute new pose to move to
        target_pose = compute_target_pose(
            pred_pose, ref_pose, p_gains, tcp_pose
        )

        # move to new pose
        robot.move_linear(target_pose[:3], target_pose[3:], quick_mode=quick_mode)

        # draw TCP frame
        robot.arm.draw_TCP(lifetime=10.0)

        # render frames
        if record_vid:
            render_img = robot.render()
            render_frames.append(render_img)

        q_key = ord("q")
        keys = robot._pb.getKeyboardEvents()
        if q_key in keys and keys[q_key] & robot._pb.KEY_WAS_TRIGGERED:
            exit()

    robot.close()

    if record_vid:
        imageio.mimwrite(
            os.path.join(os.path.dirname(__file__), "../../example_videos", "render.mp4"),
            np.stack(render_frames),
            fps=24
        )


if __name__ == '__main__':
    pass
