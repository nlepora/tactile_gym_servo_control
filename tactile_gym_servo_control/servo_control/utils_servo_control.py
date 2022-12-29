import os
import numpy as np
import torch
from torch.autograd import Variable

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tactile_gym_servo_control.learning.utils_learning import decode_pose
from tactile_gym_servo_control.learning.utils_learning import POSE_LABEL_NAMES
from tactile_gym_servo_control.robot_interface.robot_embodiment import quat2euler, euler2quat, transform, inv_transform
from tactile_gym_servo_control.utils.image_transforms import process_image


def add_gui(embodiment, init_ref_pose):

    # add user controllable ref pose to GUI
    ref_llims = [-2.0, -2.0, 2.0, -15.0, -15.0, -180.0]
    ref_ulims = [ 2.0,  2.0, 5.0,  15.0,  15.0,  180.0]

    ref_pose_ids = []
    for label_name in POSE_LABEL_NAMES:
        i = POSE_LABEL_NAMES.index(label_name)
        ref_pose_ids.append(embodiment._pb.addUserDebugParameter(label_name, 
                            ref_llims[i], ref_ulims[i], init_ref_pose[i]))

    return  ref_pose_ids


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

    print("\n Predictions: ", end="")
    predictions_arr = np.zeros(6)
    for label_name in label_names:
        predicted_val = predictions_dict[label_name].detach().cpu().numpy() 
        predictions_arr[POSE_LABEL_NAMES.index(label_name)] = predicted_val
        print(label_name, predicted_val, end=" ")

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

