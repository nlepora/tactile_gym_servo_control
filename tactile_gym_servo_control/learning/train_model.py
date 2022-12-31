"""
python train_model.py -t surface_3d
python train_model.py -t edge_2d
python train_model.py -t edge_3d
python train_model.py -t edge_5d
python train_model.py -t surface_3d edge_2d edge_3d edge_5d
"""

import os
import argparse
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tactile_gym.utils.general_utils import check_dir

from tactile_gym_servo_control.learning.utils_learning import POSE_LABEL_NAMES
from tactile_gym_servo_control.learning.utils_learning import get_pose_limits
from tactile_gym_servo_control.learning.utils_learning import encode_pose
from tactile_gym_servo_control.learning.utils_learning import decode_pose
from tactile_gym_servo_control.learning.utils_learning import acc_metric
from tactile_gym_servo_control.learning.utils_learning import err_metric
from tactile_gym_servo_control.learning.utils_learning import seed_everything
from tactile_gym_servo_control.learning.utils_plots import PlotError
from tactile_gym_servo_control.learning.utils_plots import PlotTrain

from tactile_gym_servo_control.learning.image_generator import ImageDataGenerator
from tactile_gym_servo_control.learning.setup_network import setup_network
from tactile_gym_servo_control.learning.setup_learning import setup_task
from tactile_gym_servo_control.learning.setup_learning import setup_learning
from tactile_gym_servo_control.learning.setup_learning import setup_model

data_path = os.path.join(os.path.dirname(__file__), '../../example_data/sim')
model_path = os.path.join(os.path.dirname(__file__), '../../example_models/sim')

# tolerances for accuracy metric
POS_TOL = 0.25  # mm
ROT_TOL = 1.0  # deg


def train_model(
    task,
    model,
    label_names,
    learning_params,
    image_processing_params,
    augmentation_params,
    save_dir,
    plot_during_training=True,  # slows training noticably
    device='cpu'
):

    # data dir - can specify multiple directories combined in generator
    train_data_dirs = [
        os.path.join(data_path, task, 'train')
    ]
    pose_limits = get_pose_limits(train_data_dirs, save_dir)

    validation_data_dirs = [
        os.path.join(data_path, task, 'val')
    ]

    # set generators and loaders
    generator_args = {**image_processing_params, **augmentation_params}
    train_generator = ImageDataGenerator(data_dirs=train_data_dirs, **generator_args)
    val_generator = ImageDataGenerator(data_dirs=validation_data_dirs, **image_processing_params)

    train_loader = torch.utils.data.DataLoader(
        train_generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu']
    )

    val_loader = torch.utils.data.DataLoader(
        val_generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu']
    )

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)

    # define optimizer and loss
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_params['lr'])
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=learning_params['lr_factor'],
        patience=learning_params['lr_patience'],
        verbose=True
    )

    def run_epoch(loader, n_batches, training=True):

        epoch_batch_loss = []
        epoch_batch_acc = []
        acc_df = pd.DataFrame(columns=[*POSE_LABEL_NAMES, 'overall_acc'])
        err_df = pd.DataFrame(columns=POSE_LABEL_NAMES)

        # complete dateframe of predictions and targets
        if not training:
            pred_df = pd.DataFrame(columns=POSE_LABEL_NAMES)
            targ_df = pd.DataFrame(columns=POSE_LABEL_NAMES)

        for batch in loader:

            # get inputs
            inputs, labels_dict = batch['images'], batch['labels']

            # wrap them in a Variable object
            inputs = Variable(inputs).float().to(device)

            # get labels
            labels = encode_pose(labels_dict, label_names, pose_limits, device)

            # set the parameter gradients to zero
            if training:
                optimizer.zero_grad()

            # forward pass, backward pass, optimize
            outputs = model(inputs)
            loss_size = loss(outputs, labels)

            if training:
                loss_size.backward()
                optimizer.step()

            # count correct for accuracy metric
            predictions_dict = decode_pose(outputs, label_names, pose_limits)

            if not training:
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

            # statistics
            epoch_batch_loss.append(loss_size.item())
            epoch_batch_acc.append(acc_df['overall_acc'].mean())

        # reset indices to be 0 -> test set size
        acc_df = acc_df.reset_index(drop=True).fillna(0.0)
        err_df = err_df.reset_index(drop=True).fillna(0.0)

        if not training:
            pred_df = pred_df.reset_index(drop=True).fillna(0.0)
            targ_df = targ_df.reset_index(drop=True).fillna(0.0)
            return epoch_batch_loss, epoch_batch_acc, acc_df, err_df, pred_df, targ_df
        else:
            return epoch_batch_loss, epoch_batch_acc, acc_df, err_df

    # get time for printing
    training_start_time = time.time()

    # for tracking metrics across training
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    # for saving best model
    lowest_val_loss = np.inf

    if plot_during_training:
        plot_train = PlotTrain(save_dir)
        plot_error = PlotError(save_dir, name='val_error_plot.png')

    with tqdm(total=learning_params['epochs']) as pbar:

        # Main training loop
        for epoch in range(1, learning_params['epochs'] + 1):

            train_epoch_loss, train_epoch_acc, train_acc_df, train_err_df = run_epoch(
                train_loader, n_train_batches, training=True
            )

            # ========= Validation =========
            model.eval()
            val_epoch_loss, val_epoch_acc, val_acc_df, val_err_df, val_pred_df, val_targ_df = run_epoch(
                val_loader, n_val_batches, training=False
            )
            model.train()

            # append loss and acc
            train_loss.append(train_epoch_loss)
            train_acc.append(train_epoch_acc)
            val_loss.append(val_epoch_loss)
            val_acc.append(val_epoch_acc)

            # print metrics
            print(f"\nEpoch {epoch}")
            print(f"train_acc {np.mean(train_epoch_acc):.6f}", end=', ')
            print(train_acc_df[label_names].mean().to_string().replace('\n',',   '))
            print("train_err {:.6f}".format(np.mean(train_epoch_loss)), end=', ')
            print(train_err_df[label_names].mean().to_string().replace('\n',', '))
            print(f"val_acc   {np.mean(val_epoch_acc):.6f}", end=', ')
            print(val_acc_df[label_names].mean().to_string().replace('\n',',   '))
            print("val_err   {:.6f}".format(np.mean(val_epoch_loss)), end=', ')
            print(val_err_df[label_names].mean().to_string().replace('\n',', '))

            # update plots
            if plot_during_training:
                plot_train.update(
                    train_loss, val_loss, train_acc, val_acc
                )
                plot_error.update(
                    val_pred_df, val_targ_df, val_err_df, label_names
                )

            # save the model with lowest val loss
            if np.mean(val_epoch_loss) < lowest_val_loss:
                lowest_val_loss = np.mean(val_epoch_loss)
                
                print('Saving Best Model')
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, 'best_model.pth')
                )

                # save loss and acc, save val 
                save_vars = [train_loss, val_loss, train_acc, val_acc]
                with open(os.path.join(save_dir, 'train_val_loss_acc.pkl'), 'bw') as f:  
                    pickle.dump(save_vars, f)
                
                save_vars = [val_pred_df, val_targ_df, val_err_df, label_names]
                with open(os.path.join(save_dir, 'val_pred_targ_err.pkl'), 'bw') as f:  
                    pickle.dump(save_vars, f)

            # decay the lr
            lr_scheduler.step(np.mean(val_epoch_loss))

            # update epoch progress bar
            pbar.update(1)

    print("Training finished, took {:.6f}s".format(time.time() - training_start_time))

    # save final model
    torch.save(
        model.state_dict(),
        os.path.join(save_dir, 'final_model.pth')
    )

    # final plots (note - not the best model)
    if not plot_during_training:
        plot_train = PlotTrain(save_dir, final_plot=True)
        plot_train.update(
            train_loss, val_loss, train_acc, val_acc
        )        
        plot_error = PlotError(save_dir, final_plot=True, name='final_error_plot.png')
        plot_error.update(
            val_pred_df, val_targ_df, val_err_df, label_names
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--tasks',
        nargs='+',
        help="Choose task from ['surface_3d', 'edge_2d', 'edge_3d', 'edge_5d'].",
        default=['surface_3d']
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

    for task in tasks:
        for model_type in models:

            # set save dir
            save_dir = os.path.join(model_path, model_type, task)

            # check save dir exists
            check_dir(save_dir)
            os.makedirs(save_dir, exist_ok=True)

            # setup parameters            
            network_params = setup_model(model_type, save_dir)
            learning_params, image_processing_params, augmentation_params = setup_learning(save_dir)
            out_dim, label_names = setup_task(task)            

            # create the model
            seed_everything(learning_params['seed'])

            network = setup_network(
                image_processing_params['dims'],
                out_dim,
                network_params,
                device=device
            )

            train_model(
                task,
                network,
                label_names,
                learning_params,
                image_processing_params,
                augmentation_params,
                save_dir,
                device=device        
            )
