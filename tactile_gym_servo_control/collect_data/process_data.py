import os
import shutil
import pandas as pd
import numpy as np
from tactile_gym.utils.general_utils import check_dir

data_path = os.path.join(os.path.dirname(__file__), '../../example_data/real/edge_2d_90deg')


# define split
indir_name = "data"
outdir_names = ["train", "val"]
split = 0.8

# load target df
targets_df = pd.read_csv(os.path.join(data_path, indir_name, 'targets.csv'))

# Select data
np.random.seed(0) # make predictable
inds_true = np.random.choice([True, False], size=len(targets_df), p=[split, 1-split])
inds = [inds_true, ~inds_true]

# iterate over split
for outdir_name, ind in zip(outdir_names, inds):

    indir = os.path.join(data_path, indir_name)
    outdir = os.path.join(data_path, outdir_name)

    # point image names to indir
    targets_df['sensor_image'] = f'../../{indir_name}/images/' + targets_df.sensor_image.map(str) 

    # check save dir exists
    check_dir(outdir)
    os.makedirs(outdir, exist_ok=True)

    # populate outdir
    targets_df[ind].to_csv(os.path.join(outdir, 'targets.csv'), index=False)
    file_names = ['pose_params.json', 'env_params.json', 'sensor_params.json']
    for file_name in file_names:
        shutil.copyfile(os.path.join(indir, file_name), os.path.join(outdir, file_name))
