import os
import shutil
import cv2
import pandas as pd
import numpy as np

from tactile_gym.utils.general_utils import check_dir
from tactile_gym.utils.general_utils import save_json_obj, load_json_obj
from tactile_gym_servo_control.utils.image_transforms import process_image

data_path = os.path.join(os.path.dirname(__file__), '../../example_data/real')

tasks = ['edge_2d']#, 'surface_2d']

# define split
indir_name = "data"
outdir_names = ["train", "val"]
split = 0.8

# optional image processing
sensor_params = {
    'thresh': True,
    'dims': (128,128),
    "circle_mask_radius": 150,
    "bbox": [10, 10, 310, 310]
    }

for task in tasks:

    # load target df
    targets_df = pd.read_csv(os.path.join(data_path, task, indir_name, 'targets.csv'))

    # Select data
    np.random.seed(0) # make predictable
    inds_true = np.random.choice([True, False], size=len(targets_df), p=[split, 1-split])
    inds = [inds_true, ~inds_true]

    # iterate over split
    for outdir_name, ind in zip(outdir_names, inds):

        indir = os.path.join(data_path, task, indir_name)
        outdir = os.path.join(data_path, task, outdir_name)

        # check save dir exists
        check_dir(outdir)
        os.makedirs(outdir, exist_ok=True)

        # copy param files
        shutil.copy(os.path.join(indir, 'pose_params.json'), outdir)
        shutil.copy(os.path.join(indir, 'env_params.json'), outdir)
        
        # optional - process image data if sensor_params supplied
        try:
            sensor_params_in = load_json_obj(os.path.join(indir, 'sensor_params'))
            sensor_params_out = {**sensor_params_in, **sensor_params}
            
            if 'bbox' in sensor_params_in.keys() and 'bbox' in sensor_params.keys():
                b, b_in = sensor_params['bbox'], sensor_params_in['bbox']
                sensor_params_out['bbox'] = [b_in[0]+b[0], b_in[1]+b[1], b_in[0]+b[2], b_in[1]+b[3]]

            save_json_obj(sensor_params_out, os.path.join(outdir, 'sensor_params'))
            
            image_dir = os.path.join(outdir, "images")
            os.makedirs(image_dir, exist_ok=True)

            for img_name in targets_df[ind].sensor_image:
                print(f'processed {outdir_name}: {img_name}')
                infile = os.path.join(indir, 'images', img_name)
                outfile = os.path.join(outdir, 'images', img_name)
                img = cv2.imread(infile)
                proc_img = process_image(img, **sensor_params)
                cv2.imwrite(outfile, proc_img)
        
        # except just point to original data
        except:
            shutil.copy(os.path.join(indir, 'sensor_params.json'), outdir)  

            targets_df.loc[ind,'sensor_image'] = \
                f'../../{indir_name}/images/' + targets_df[ind].sensor_image.map(str)
        
        # save targets
        targets_df[ind].to_csv(os.path.join(outdir, 'targets.csv'), index=False)
