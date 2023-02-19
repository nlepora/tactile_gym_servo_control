import os
import tkinter as tk
import numpy as np
import torch
from torch.autograd import Variable

try:
    import pyspacemouse
except:
    print('pyspacemouse not installed')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tactile_gym_servo_control.utils.utils_learning import decode_pose
from tactile_gym_servo_control.utils.utils_learning import POSE_LABEL_NAMES
from tactile_gym_servo_control.utils.image_transforms import process_image

LEFT, RIGHT, FORE, BACK, SHIFT, CTRL, QUIT \
    = 65295, 65296, 65297, 65298, 65306, 65307, ord('Q')


class ManualControl:
    def __init__(self, embodiment):    
        self.embodiment = embodiment
        try:
            pyspacemouse.open()
        except:
            print('no spacemouse')

    def keyboard(self,
        delta_init = [0.0, 0, 0, 0, 0, 0]
    ):
        delta = np.array([0, 0, 0, 0, 0, 0]) + delta_init # stop keeping state

        keys = self.embodiment.controller._client._sim_env._pb.getKeyboardEvents()
        if CTRL in keys:
            if FORE in keys:  delta -= [0, 0, 0, 0, 1, 0]
            if BACK in keys:  delta += [0, 0, 0, 0, 1, 0]
            if RIGHT in keys: delta -= [0, 0, 0, 1, 0, 0]
            if LEFT in keys:  delta += [0, 0, 0, 1, 0, 0]
        elif SHIFT in keys:
            if FORE in keys:  delta -= [0, 0, 1, 0, 0, 0]
            if BACK in keys:  delta += [0, 0, 1, 0, 0, 0]
            if RIGHT in keys: delta -= [0, 0, 0, 0, 0, 2.5]
            if LEFT in keys:  delta += [0, 0, 0, 0, 0, 2.5]
        else:
            if FORE in keys:  delta -= [1, 0, 0, 0, 0, 0]
            if BACK in keys:  delta += [1, 0, 0, 0, 0, 0]
            if RIGHT in keys: delta -= [0, 1, 0, 0, 0, 0]
            if LEFT in keys:  delta += [0, 1, 0, 0, 0, 0]
        if QUIT in keys:  delta = None

        return np.array(delta)

    def spacemouse(self,
        delta_init = [0, 0, 0, 0, 0, 0], 
        gain = 2, 
        sign = [-1, -1, -1, -1, 1, -1]
    ):
        state = pyspacemouse.read()
        pose_state = np.array([state[i]*sign[i-1] for i in [2, 1, 3, 4, 5, 6]])
        delta = delta_init + gain * pose_state

        return delta


class Slider:
    def __init__(self,
        init_pose,
        pose_llims=[-5, -5, 0, -15, -15, -180],
        pose_ulims=[ 5,  5, 5,  15,  15,  180]
    ):    
        self.pose_ids = []
        self.tk = tk.Tk()
        self.tk.geometry("300x500+200+0")
        for i, label_name in enumerate(POSE_LABEL_NAMES):
            self.pose_ids.append(
                tk.Scale(self.tk, from_=pose_llims[i], to=pose_ulims[i],
                    label=label_name, length=400, orient=tk.HORIZONTAL, 
                    tickinterval=(pose_ulims[i]-pose_llims[i])/4, resolution=0.1
                )
            )
            self.pose_ids[i].pack()
            self.pose_ids[i].set(init_pose[i])

    def read(self, pose):
        self.tk.update_idletasks()
        self.tk.update()
        for i in range(len(pose)):
            pose[i] = self.pose_ids[i].get()
        return pose


class Slider_sim:
    def __init__(self,
        embodiment, init_pose,
        pose_llims=[-5, -5, 0, -15, -15, -180],
        pose_ulims=[ 5,  5, 5,  15,  15,  180]
    ):    
        self.embodiment = embodiment
        self.pose_ids = []
        for i, label_name in enumerate(POSE_LABEL_NAMES):
            self.pose_ids.append(
                embodiment.controller._client._sim_env._pb.addUserDebugParameter(
                    label_name, pose_llims[i], pose_ulims[i], init_pose[i]
                )
            )

    def read(self, pose):
        for i in range(len(pose)):
            pose[i] = self.embodiment.controller._client._sim_env._pb.readUserDebugParameter(
                self.pose_ids[i]
            ) 
        return pose


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
            with np.printoptions(precision=1, suppress=True):
                print(label_name, predicted_val, end="")

        return predictions_arr


