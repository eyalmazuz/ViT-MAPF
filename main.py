import pandas as pd

from train import train_xgb, train_vit, train_vivit
from utils import seed_everything

# Training settings
hparams = {
    "batch_size": 512,
    "epochs": 5,
    "lr": 3e-5,
    "gamma": 0.7,
    "image_size": 256,
    "patch_size": 16,
    "dim": 32,
    "depth": 2,
    "heads": 4,
    "mlp_dim": 128,
    "frame_patch_size": 1,
    "temporal_depth": 6
}

seed_everything(42)

images_path = "./data_frames_map_paths_start_goal_agg"

df = pd.read_csv("./MovingAIData-labelled-with-features.csv",) # usecols=['GridName', 'InstanceId', 'problem_type', 'NumOfAgents', 'Y',
                                                                    #   'sat Success', 'icts Success', 'cbsh-c Success', 'lazycbs Success', 'epea Success',
                                                                    #   'sat Runtime', 'icts Runtime', 'cbsh-c Runtime', 'lazycbs Runtime', 'epea Runtime'])

df['GridColumns'] = 256
df['GridRows'] = 256
df['GridSize'] = 256 * 256

df['path'] = df['GridName'] + '-' + df['problem_type'] + '-' + df['InstanceId'].astype(str) + '-' + df['NumOfAgents'].astype(str) + '.npz'

#train_xgb(df, 'random')
#train_vit(df, "random", images_path, hparams)
train_vivit(df, 'random', images_path, hparams)
