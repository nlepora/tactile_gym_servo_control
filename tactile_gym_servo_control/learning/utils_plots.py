import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tactile_gym_servo_control.learning.utils_learning import POSE_LABEL_NAMES

sns.set_theme(style="darkgrid")
model_path = os.path.join(os.path.dirname(__file__), '../../example_models')


class PlotError:
    def __init__(self,
        save_dir=None, final_plot=False, name="error_plot.png"
    ):
        self._save_dir = save_dir
        self._final_plot = final_plot
        self._name = name
        
        if not final_plot:
            plt.ion()
        
        self._fig, self._axs = plt.subplots(2, 3, figsize=(12, 7))
        self._fig.subplots_adjust(wspace=0.3)

    def update(self, 
        pred_df, targ_df, err_df, label_names
    ):
        
        if not self._final_plot:
            for ax in self._axs.flat:
                ax.clear()

        n_smooth = int(pred_df.shape[0] / 20)

        for i, ax in enumerate(self._axs.flat):

            pose_label = POSE_LABEL_NAMES[i]
            if pose_label not in label_names:
                continue

            # sort all dfs by target
            targ_df = targ_df.sort_values(by=pose_label)

            pred_df = pred_df.assign(temp=targ_df[pose_label])
            pred_df = pred_df.sort_values(by='temp')
            pred_df = pred_df.drop('temp', axis=1)

            err_df = err_df.assign(temp=targ_df[pose_label])
            err_df = err_df.sort_values(by='temp')
            err_df = err_df.drop('temp', axis=1)

            ax.scatter(
                targ_df[pose_label], pred_df[pose_label], s=1,
                c=err_df[pose_label], cmap="inferno"
            )
            ax.plot(
                targ_df[pose_label].rolling(n_smooth).mean(),
                pred_df[pose_label].rolling(n_smooth).mean(),
                linewidth=2, c='r'
            )
            ax.set(xlabel=f"target {pose_label}", ylabel=f"predicted {pose_label}")

            pose_llim = np.round(min(targ_df[pose_label]))
            pose_ulim = np.round(max(targ_df[pose_label]))
            ax.set_xlim(pose_llim, pose_ulim)
            ax.set_ylim(pose_llim, pose_ulim)

            ax.text(0.05, 0.9, 'MAE = {:.4f}'.format(err_df[pose_label].mean()), transform=ax.transAxes)
            ax.grid(True)

        if self._save_dir is not None:
            save_file = os.path.join(self._save_dir, self._name)
            self._fig.savefig(save_file, dpi=320, pad_inches=0.01, bbox_inches='tight')

        if self._final_plot:
            plt.show()
        else:
            self._fig.canvas.draw()
            plt.pause(0.01)


class PlotTrain:
    def __init__(self,
        save_dir=None, final_plot=False, name="train_plot.png"
    ):
        self._save_dir = save_dir
        self._final_plot = final_plot
        self._name = name

        if not final_plot:
            plt.ion()

        self._fig, self._axs = plt.subplots(1, 2, figsize=(12, 4))
 
    def update(self, 
        train_loss, val_loss, train_acc, val_acc
    ):
        for ax in self._axs.flat:
            ax.clear()
            
        # convert lists to arrays
        train_loss = np.array(train_loss)
        val_loss = np.array(val_loss)
        train_acc = np.array(train_acc)
        val_acc = np.array(val_acc)

        n_epochs = train_loss.shape[0]
        r_epochs = np.arange(1, n_epochs+1)

        for loss, color in zip([train_loss, val_loss], ['r', 'b']):
            lo_bound = np.clip(loss.mean(axis=1) - loss.std(axis=1), loss.min(axis=1), loss.max(axis=1))
            up_bound = np.clip(loss.mean(axis=1) + loss.std(axis=1), loss.min(axis=1), loss.max(axis=1))
            self._axs[0].plot(r_epochs, loss.mean(axis=1), color=color, alpha=1.0)
            self._axs[0].fill_between(r_epochs, lo_bound, up_bound, color=color, alpha=0.25)

        self._axs[0].set_yscale('log')
        self._axs[0].set_xlabel('Epoch')
        self._axs[0].set_ylabel('Loss')

        for acc, color in zip([train_acc, val_acc], ['r', 'b']):
            lo_bound = np.clip(acc.mean(axis=1) - acc.std(axis=1), acc.min(axis=1), acc.max(axis=1))
            up_bound = np.clip(acc.mean(axis=1) + acc.std(axis=1), acc.min(axis=1), acc.max(axis=1))
            self._axs[1].plot(r_epochs, acc.mean(axis=1), color=color, alpha=1.0)
            self._axs[1].fill_between(r_epochs, lo_bound, up_bound, color=color, alpha=0.25, label='_nolegend_')

        self._axs[1].set_xlabel('Epoch')
        self._axs[1].set_ylabel('Accuracy')
        plt.legend(['Train', 'Val'])

        if self._save_dir is not None:
            save_file = os.path.join(self._save_dir, self._name)
            self._fig.savefig(save_file, dpi=320, pad_inches=0.01, bbox_inches='tight')

        if not self._final_plot:
            for ax in self._axs.flat:
                ax.relim()
                ax.set_xlim([0, n_epochs])
                ax.autoscale_view(True, True, True)

        if self._final_plot:
            plt.show()
        else:
            self._fig.canvas.draw()
            plt.pause(0.01)


if __name__ == '__main__':

    task = 'surface_3d'
    # task = 'edge_2d'
    # task = 'edge_3d'
    # task = 'edge_5d'

    model = 'simple_cnn'

    # path to model for loading
    save_dir = os.path.join(model_path, model, task)

    # load and plot predictions
    with open(os.path.join(save_dir, 'val_pred_targ_err.pkl'), 'rb') as f: 
        pred_df, targ_df, err_df, label_names = pickle.load(f)

    plot_error = PlotError(save_dir, final_plot=True, name='val_error_plot.png')
    plot_error.update(pred_df, targ_df, err_df, label_names)
    
    # load and plot training
    with open(os.path.join(save_dir, 'train_val_loss_acc.pkl'), 'rb') as f: 
        train_loss, val_loss, train_acc, val_acc = pickle.load(f)

    plot_train = PlotTrain(save_dir, final_plot=True)
    plot_train.update(train_loss, val_loss, train_acc, val_acc)
