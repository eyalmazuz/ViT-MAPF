import pandas as pd

from train import train_xgb, train_vit
from utils import seed_everything

# Training settings
hparams = {
    "batch_size": 128,
    "epochs": 20,
    "lr": 3e-5,
    "gamma": 0.7,
    "image_size": 256,
    "patch_size": 32,
    "dim": 1024,
    "depth": 24,
    "heads": 16,
    "mlp_dim": 4096
}

seed_everything(42)

images_path = "./data_map_start_goal_paths"

df = pd.read_csv("./MovingAIData-labelled-with-features.csv",) # usecols=['GridName', 'InstanceId', 'problem_type', 'NumOfAgents', 'Y',
                                                                    #   'sat Success', 'icts Success', 'cbsh-c Success', 'lazycbs Success', 'epea Success',
                                                                    #   'sat Runtime', 'icts Runtime', 'cbsh-c Runtime', 'lazycbs Runtime', 'epea Runtime'])

df['path'] = df['GridName'] + '-' + df['problem_type'] + '-' + df['InstanceId'].astype(str) + '-' + df['NumOfAgents'].astype(str) + '.npz'

# train_xgb(df, 'map_type')
train_vit(df, "map_type", images_path, hparams)