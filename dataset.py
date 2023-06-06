import os

import numpy as np
import torch
from torch.utils.data import Dataset


alg2label = {
    'sat Runtime': 0,
    'icts Runtime': 1,
    'cbsh-c Runtime': 2,
    'lazycbs Runtime': 3,
    'epea Runtime': 4, 
}

success_order = ['sat Success', 'icts Success', 'cbsh-c Success', 'lazycbs Success', 'epea Success']
runtime_order = ['sat Runtime', 'icts Runtime', 'cbsh-c Runtime', 'lazycbs Runtime', 'epea Runtime']

class MAPFDataset(Dataset):
    def __init__(self, image_path, df, transform=None):
        self.image_path = image_path
        self.df = df.groupby(['GridName', 'problem_type', 'InstanceId', 'NumOfAgents']).first()
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_path, self.df.iloc[idx].path)
        img = torch.from_numpy(np.load(img_path)['arr_0']).permute(2, 0, 1).to(torch.float32)

        if self.transform:
            img_transformed = self.transform(img)

        *GridName, problem_type, InstanceId, NumOfAgents = img_path.split('/')[-1][:-4].split('-')
        
        fastest_algorthim = self.df.loc[('-'.join(GridName), problem_type, int(InstanceId), int(NumOfAgents))].Y
        label = alg2label[fastest_algorthim]

        success_vector = self.df.loc[('-'.join(GridName), problem_type, int(InstanceId), int(NumOfAgents))][success_order].values.astype(np.int64)
        runtime_vector = self.df.loc[('-'.join(GridName), problem_type, int(InstanceId), int(NumOfAgents))][runtime_order].values.astype(np.int64)
        
        return img_transformed, label, torch.from_numpy(success_vector), torch.from_numpy(runtime_vector)


class XGBoostMAPFDataset(Dataset):

    feature_columns = ['GridColumns', 'GridRows', 'GridSize', 'NumOfObstacles', 'ObstacleDensity',
       'AvgDistanceToGoal', 'MaxDistanceToGoal', 'MinDistanceToGoal',
       'AvgStartDistances', 'AvgGoalDistances', 'PointsAtSPRatio', 'Sparsity',
       '2waycollisions', 'StdDistanceToGoal', 'MaxStartDistances',
       'MaxGoalDistances', 'MinStartDistances', 'MinGoalDistances',
       'StdStartDistances', 'StdGoalDistances']

    def __init__(self, df):
        self.df = df.groupby(['GridName', 'problem_type', 'InstanceId', 'NumOfAgents']).first()

    def __len__(self):
        return 1
    
    def __getitem__(self, idx):

        features = self.df.loc[:, self.feature_columns]

        fastest_algorthim = self.df.Y
        label = fastest_algorthim.apply(lambda alg: alg2label[alg])

        success_vector = self.df.loc[:, success_order].values.astype(np.int64)
        runtime_vector = self.df.loc[:, runtime_order].values.astype(np.int64)

        return features, label, success_vector, runtime_vector
    

class ViViTMAPFDataset(Dataset):
    def __init__(self, image_path, df, transform=None):
        self.image_path = image_path
        self.df = df.groupby(['GridName', 'problem_type', 'InstanceId', 'NumOfAgents']).first()
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_path, self.df.iloc[idx].path)
        img = torch.from_numpy(np.load(img_path)['arr_0']).permute(2, 0, 1).to(torch.float32)

        if self.transform:
            img_transformed = self.transform(img)

        *GridName, problem_type, InstanceId, NumOfAgents = img_path.split('/')[-1][:-4].split('-')
        
        fastest_algorthim = self.df.loc[('-'.join(GridName), problem_type, int(InstanceId), int(NumOfAgents))].Y
        label = alg2label[fastest_algorthim]

        success_vector = self.df.loc[('-'.join(GridName), problem_type, int(InstanceId), int(NumOfAgents))][success_order].values.astype(np.int64)
        runtime_vector = self.df.loc[('-'.join(GridName), problem_type, int(InstanceId), int(NumOfAgents))][runtime_order].values.astype(np.int64)
        
        return img_transformed, label, torch.from_numpy(success_vector), torch.from_numpy(runtime_vector)


def main():
    pass

if __name__ == "__main__":
    main() 
