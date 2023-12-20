import os
import sys
import pandas as pd

from train import train_xgb, train_vit, train_vivit, train_vit_ensemble, train_vivit_ensemble
from utils import seed_everything

if 'SLURM_ARRAY_TASK_ID' in os.environ:
    test_set_number = int(os.environ['SLURM_ARRAY_TASK_ID'])
else:
    test_set_number = None

# Training settings
hparams = {
    "batch_size": 512,
    "epochs": 10,
    "lr": 3e-5,
    "gamma": 0.7,
    "image_size": 256,
    "patch_size": 16,
    "dim": 128,
    "depth": 4,
    "heads": 8,
    "mlp_dim": 128,
    "frame_patch_size": 1,
    "temporal_depth": 6
}

seed_everything(42)

images_path = "/sise/bshapira-group/mazuze-davidyu/ViT-MAPF/data_frames_map_paths_start_goal_agg"
# images_path = "./custom_mapf_images_all_paths"
df = pd.read_csv("/sise/bshapira-group/mazuze-davidyu/ViT-MAPF/MovingAIData-labelled-with-features.csv",)
# df = pd.read_csv("./MovingAIData-labelled-with-features.csv")


# df['GridColumns'] = 256
# df['GridRows'] = 256
# df['GridSize'] = 256 * 256

df['path'] = df['GridName'] + '-' + df['problem_type'] + '-' + df['InstanceId'].astype(str) + '-' + df['NumOfAgents'].astype(str) + '.npz'

print(df.shape)
ex = os.listdir(images_path)
df = df[df.path.isin(ex)]

print(df.shape)
# train_xgb(df, 'map_type', test_set_number=test_set_number)
# train_vit(df, "map_type", images_path, hparams, test_set_number=test_set_number)
# train_vivit(df, 'map_type', images_path, hparams, test_set_number=test_set_number)
train_vit_ensemble(df, 'map_type', images_path, hparams, test_set_number=test_set_number)
train_vivit_ensemble(df, 'map_type', images_path, hparams, test_set_number=test_set_number)
