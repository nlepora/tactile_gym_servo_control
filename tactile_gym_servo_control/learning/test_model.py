"""
python test_cnn.py -t surface_3d
python test_cnn.py -t edge_2d
python test_cnn.py -t edge_3d
python test_cnn.py -t edge_5d
python test_cnn.py -t surface_3d edge_2d edge_3d edge_5d
"""
import os
import argparse
import pandas as pd
from torch.autograd import Variable
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tactile_gym.utils.general_utils import load_json_obj

from tactile_gym_servo_control.learning.utils_learning import POSE_LABEL_NAMES
from tactile_gym_servo_control.learning.utils_learning import decode_pose
from tactile_gym_servo_control.learning.utils_learning import acc_metric
from tactile_gym_servo_control.learning.utils_learning import err_metric
from tactile_gym_servo_control.learning.utils_plots import PlotError

from tactile_gym_servo_control.learning.image_generator import ImageDataGenerator
from tactile_gym_servo_control.learning.setup_network import setup_network
from tactile_gym_servo_control.learning.setup_learning import setup_task

data_path = os.path.join(os.path.dirname(__file__), '../../example_data/real')
model_path = os.path.join(os.path.dirname(__file__), '../../example_models/real')

# tolerances for accuracy metric
POS_TOL = 0.25  # mm
ROT_TOL = 1.0  # deg


def test_model(
    task,
    model,
    label_names,
    pose_limits,
    learning_params,
    image_processing_params,
    save_dir,
    device='cpu'
):

    # data dir (can specifiy multiple directories combined in generator)
    test_data_dirs = [
        os.path.join(data_path, task, 'val')
    ]

    # set generators and loaders
    test_generator = ImageDataGenerator(data_dirs=test_data_dirs, **image_processing_params)

    test_loader = torch.utils.data.DataLoader(
        test_generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu']
    )

    # complete dateframe of predictions and targets
    pred_df = pd.DataFrame(columns=POSE_LABEL_NAMES)
    targ_df = pd.DataFrame(columns=POSE_LABEL_NAMES)

    # complete dateframe of accuracy and errors
    acc_df = pd.DataFrame(columns=[*POSE_LABEL_NAMES, 'overall_acc'])
    err_df = pd.DataFrame(columns=POSE_LABEL_NAMES)

    for i, batch in enumerate(test_loader):

        # get inputs
        inputs, labels_dict = batch['images'], batch['labels']

        # wrap them in a Variable object
        inputs = Variable(inputs).float().to(device)

        # forward pass
        outputs = model(inputs)

        # count correct for accuracy metric
        predictions_dict = decode_pose(outputs, label_names, pose_limits)

        # append predictions and labels to dataframes
        batch_pred_df = pd.DataFrame.from_dict(predictions_dict)
        batch_targ_df = pd.DataFrame.from_dict(labels_dict)
        pred_df = pd.concat([pred_df, batch_pred_df])
        targ_df = pd.concat([targ_df, batch_targ_df])

        # get errors and accuracy
        batch_err_df = err_metric(labels_dict, predictions_dict, label_names)
        batch_acc_df = acc_metric(batch_err_df, label_names, POS_TOL, ROT_TOL)

        # append error to dataframe
        err_df = pd.concat([err_df, batch_err_df])
        acc_df = pd.concat([acc_df, batch_acc_df])

    # reset indices to be 0 -> test set size
    pred_df = pred_df.reset_index(drop=True).fillna(0.0)
    targ_df = targ_df.reset_index(drop=True).fillna(0.0)
    acc_df = acc_df.reset_index(drop=True).fillna(0.0)
    err_df = err_df.reset_index(drop=True).fillna(0.0)

    print("Test Metrics")
    print("test_acc:")
    print(acc_df[[*label_names, 'overall_acc']].mean())
    print("test_err:")
    print(err_df[label_names].mean())

    # plot full error graph
    plot_error = PlotError(save_dir, final_plot=True, name='test_error_plot.png')
    plot_error.update(
        pred_df, targ_df, err_df, label_names
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--tasks',
        nargs='+',
        help="Choose task from ['surface_3d', 'edge_2d', 'edge_3d', 'edge_5d'].",
        default=['edge_2d']
    )
    parser.add_argument(
        '-m', '--models',
        nargs='+',
        help="Choose model from ['simple_cnn', 'nature_cnn', 'resnet', 'vit'].",
        default=['simple_cnn']
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
    models = args.models
    device = args.device
    version = ''

    # test the trained networks
    for model_type in models:
        for task in tasks:

            # task specific parameters
            out_dim, label_names = setup_task(task)

            # set save dir
            task += version
            save_dir = os.path.join(model_path, model_type, task)

            # setup parameters            
            network_params = load_json_obj(os.path.join(save_dir, 'model_params'))
            learning_params = load_json_obj(os.path.join(save_dir, 'learning_params'))
            image_processing_params = load_json_obj(os.path.join(save_dir, 'image_processing_params'))

            # get the pose limits used for encoding/decoding pose/predictions
            pose_params = load_json_obj(os.path.join(save_dir, 'pose_params'))
            pose_limits = [pose_params['pose_llims'], pose_params['pose_ulims']]

            # create the model
            network = setup_network(
                image_processing_params['dims'],
                out_dim,
                network_params,
                saved_model_dir=save_dir,
                device=device
            )
            network.eval()

            test_model(
                task,
                network,
                label_names,
                pose_limits,
                learning_params,
                image_processing_params,
                save_dir,
                device=device
            )
