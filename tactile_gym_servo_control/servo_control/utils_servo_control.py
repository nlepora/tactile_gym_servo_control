import os
import numpy as np
import torch
from torch.autograd import Variable

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tactile_gym_servo_control.learning.utils_learning import decode_pose
from tactile_gym_servo_control.learning.utils_learning import POSE_LABEL_NAMES
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


class Model:
    def __init__(self,
        model, image_processing_params, pose_params, 
        label_names, 
        device='cpu'
    ):
        self.model = model
        self.image_processing_params = image_processing_params
        self.label_names = label_names 
        self.pose_limits = [pose_params['pose_llims'], pose_params['pose_ulims']]
        self.device = device

    def predict(self, 
        tactile_image
    ):
        
        processed_image = process_image(
            tactile_image,
            gray=False,
            **self.image_processing_params
        )

        # channel first for pytorch
        processed_image = np.rollaxis(processed_image, 2, 0)

        # add batch dim
        processed_image = processed_image[np.newaxis, ...]

        # perform inference with the trained model
        model_input = Variable(torch.from_numpy(processed_image)).float().to(self.device)
        raw_predictions = self.model(model_input)

        # decode the prediction
        predictions_dict = decode_pose(raw_predictions, self.label_names, self.pose_limits)

        print("\n Predictions: ", end="")
        predictions_arr = np.zeros(6)
        for label_name in self.label_names:
            predicted_val = predictions_dict[label_name].detach().cpu().numpy() 
            predictions_arr[POSE_LABEL_NAMES.index(label_name)] = predicted_val
            print(label_name, predicted_val, end=" ")

        return predictions_arr
