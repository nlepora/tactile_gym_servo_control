import os
import numpy as np
import torch
from torch.autograd import Variable

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tactile_gym_servo_control.learning.learning_utils import decode_pose
from tactile_gym_servo_control.learning.learning_utils import POSE_LABEL_NAMES, POS_LABEL_NAMES, ROT_LABEL_NAMES
from tactile_gym_servo_control.cri_wrapper.cri_embodiment import quat2euler, euler2quat, transform, inv_transform
from tactile_gym_servo_control.utils.image_transforms import process_image


def add_gui(embodiment, init_ref_pose):

    # add user controllable ref pose to GUI
    ref_pose_ids = []
    ref_pose_ids.append(embodiment._pb.addUserDebugParameter('x', -2.0, 2.0, init_ref_pose[POSE_LABEL_NAMES.index('x')]))
    ref_pose_ids.append(embodiment._pb.addUserDebugParameter('y', -2.0, 2.0, init_ref_pose[POSE_LABEL_NAMES.index('y')]))
    ref_pose_ids.append(embodiment._pb.addUserDebugParameter('z', 2.0, 5.0, init_ref_pose[POSE_LABEL_NAMES.index('z')]))
    ref_pose_ids.append(embodiment._pb.addUserDebugParameter('Rx', -15.0, 15.0, init_ref_pose[POSE_LABEL_NAMES.index('Rx')]))
    ref_pose_ids.append(embodiment._pb.addUserDebugParameter('Ry', -15.0, 15.0, init_ref_pose[POSE_LABEL_NAMES.index('Ry')]))
    ref_pose_ids.append(embodiment._pb.addUserDebugParameter('Rz', -180.0, 180.0, init_ref_pose[POSE_LABEL_NAMES.index('Rz')]))

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

