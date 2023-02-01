import os

from tactile_gym.utils.general_utils import save_json_obj


def setup_learning(save_dir):

    # Parameters
    learning_params = {
        'seed': 42,
        'batch_size': 128,
        'epochs': 100,
        'lr': 1e-4,
        'lr_factor': 0.5,
        'lr_patience': 10,
        'shuffle': True,
        'n_cpu': 1
    }

    image_processing_params = {
        'dims': (128, 128),
        'bbox': None,
        'thresh': False,
        'stdiz': False,
        'normlz': True,
    }

    augmentation_params = {
        'rshift': (0.025, 0.025),
        'rzoom': None,
        'brightlims': None,
        'noise_var': None,
    }

    save_json_obj(learning_params, os.path.join(save_dir, 'learning_params'))
    save_json_obj(image_processing_params, os.path.join(save_dir, 'image_processing_params'))
    save_json_obj(augmentation_params, os.path.join(save_dir, 'augmentation_params'))

    return learning_params, image_processing_params, augmentation_params


def setup_model(model_type, save_dir):
    
    model_params = {
        'model_type': model_type
    }

    if model_type == 'simple_cnn':
        model_params['model_kwargs'] = {
                'conv_layers': [16, 32, 32, 32],
                'conv_kernel_sizes': [5, 5, 5, 5],
                'fc_layers': [512, 512],
                'dropout': 0.0,
                'apply_batchnorm': True,
        }

    elif model_type == 'nature_cnn':
        model_params['model_kwargs'] = {
            'fc_layers': [512, 512],
            'dropout': 0.0,
        }

    elif model_type == 'resnet':
        model_params['model_kwargs'] = {
            'layers': [2, 2, 2, 2],
        }

    elif model_type == 'vit':
        model_params['model_kwargs'] = {
            'patch_size': 32,
            'dim': 128,
            'depth': 6,
            'heads': 8,
            'mlp_dim': 512,
        }

    # save parameters
    save_json_obj(model_params, os.path.join(save_dir, 'model_params'))

    return model_params


def setup_task(task_name):
    """
    Returns task specific details.
    """

    if task_name == 'surface_3d':
        out_dim = 5
        label_names = ['z', 'Rx', 'Ry']

    elif task_name == 'edge_2d':
        out_dim = 3
        label_names = ['x', 'Rz']

    elif task_name == 'surface_2d':
        out_dim = 3
        label_names = ['y', 'Rz']

    elif task_name == 'edge_3d':
        out_dim = 4
        label_names = ['y', 'z', 'Rz']

    elif task_name == 'edge_5d':
        out_dim = 8
        label_names = ['y', 'z', 'Rx', 'Ry', 'Rz']

    elif task_name == 'pushing_2d':
        out_dim = 3
        label_names = ['y', 'Rz']

    elif task_name == 'pushing_3d':
        out_dim = 5
        label_names = ['z', 'Rx', 'Ry']
        
    else:
        raise ValueError('Incorrect task_name specified: {}'.format(task_name))

    return out_dim, label_names
