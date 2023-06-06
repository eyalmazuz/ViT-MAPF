import pandas as pd

import wandb

from train import train_xgb, train_vit
from utils import seed_everything

# Training settings
batch_size = 128
epochs = 20
lr = 3e-5
gamma = 0.7
seed = 42


run = wandb.init(
    # Set the project where this run will be logged
    project="MAPF-ViT",
    # Track hyperparameters and run metadata
    config={
        'batch_size': 128,
        'epochs': 20,
        'lr': 3e-5,
        'gamma': 0.7,
        'seed': 42,
    })

seed_everything(seed)

df = pd.read_csv('./MovingAIData-labelled-with-features.csv',) # usecols=['GridName', 'InstanceId', 'problem_type', 'NumOfAgents', 'Y',
                                                                    #   'sat Success', 'icts Success', 'cbsh-c Success', 'lazycbs Success', 'epea Success',
                                                                    #   'sat Runtime', 'icts Runtime', 'cbsh-c Runtime', 'lazycbs Runtime', 'epea Runtime'])

train_vit(df, 'map_type', run)