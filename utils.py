import os
import random

import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def calc_fastest(preds, labels):
    fastest_accuracy = (preds.argmax(dim=1) == labels).float().mean()
    return fastest_accuracy

def calc_coverage(preds ,successes):
    coverage_accuracy = (successes[torch.arange(successes.shape[0]), preds.argmax(dim=1)] == 1).float().mean()
    return coverage_accuracy

def calc_coverage_runtime(preds ,successes):
    coverage_accuracy = (successes[torch.arange(successes.shape[0]), preds.argmax(dim=1)] < 300000).float().mean()
    return coverage_accuracy
    

def get_random_split(df):
    rskf = RepeatedKFold(n_repeats=2, n_splits=5)

    for i, (train_index, test_index) in enumerate(rskf.split(df)):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]

        yield train_df, test_df
        

def get_stratified_random_split(df):
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42, stratify= df['Y'])
    
    yield train_df, test_df
    


def get_map_type_split(df, n=1, test_set_number=None):
    map_types = df.maptype.unique().tolist()
    # random.shuffle(map_types)

    splits = [map_types[i:i+n] for i in range(0, len(map_types), n)]
    print(splits)
    print(f"test_set_number={test_set_number}")

    if test_set_number is not None:
        print(f'using split {splits[test_set_number]}')
        train_df = df[~df.maptype.isin(splits[test_set_number])]
        test_df = df[df.maptype.isin(splits[test_set_number])]

        print(f'train_df.maptype.unique()={train_df.maptype.unique()}')
        print(f'test_df.maptype.unique()={test_df.maptype.unique()}')

        yield train_df, test_df, splits[test_set_number]

    else:
        for split in splits:
            train_df = df[~df.maptype.isin(split)]
            test_df = df[df.maptype.isin(split)]

            print(f'train_df.maptype.unique()={train_df.maptype.unique()}')
            print(f'test_df.maptype.unique()={test_df.maptype.unique()}')

            yield train_df, test_df

def get_grid_name_split(df, n=5):
    grid_names = df.GridName.unique()
    random.shuffle(grid_names)

    splits = [grid_names[i:i+n] for i in range(0, len(grid_names), n)]

    for split in splits:
        train_df = df[~df.GridName.isin(split)]
        test_df = df[df.GridName.isin(split)]

        yield train_df, test_df


def get_split(df, split_type: str, test_set_number=None):
    if split_type == 'random':
        return get_stratified_random_split(df)
    
    elif split_type == 'map_type':
        return get_map_type_split(df, test_set_number=test_set_number)

    elif split_type == 'grid_name':
        return get_grid_name_split(df)

    else:
        raise ValueError
