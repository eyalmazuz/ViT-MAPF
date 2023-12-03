import os

import numpy as np
import torch
from tqdm.auto import tqdm
from xgboost import XGBClassifier
from wandb.xgboost import WandbCallback

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda import amp
torch.backends.cudnn.benchmark = True

from vit_pytorch import SimpleViT
from vit_pytorch.vivit import ViT as ViVit

import wandb

from dataset import MAPFDataset, ViViTMAPFDataset

from utils import calc_coverage, calc_coverage_runtime, calc_fastest, get_split, seed_everything

alg2label = {
    'sat Runtime': 0,
    'icts Runtime': 1,
    'cbsh-c Runtime': 2,
    'lazycbs Runtime': 3,
    'epea Runtime': 4, 
}


def train_xgb(df, split_type, test_set_number=None):

    run = wandb.init(
    # Set the project where this run will be logged
    project="MAPF-ViT",
    # Track hyperparameters and run metadata
    tags=['XGBoost'])

    success_order = ['sat Success', 'icts Success', 'cbsh-c Success', 'lazycbs Success', 'epea Success']
    runtime_order = ['sat Runtime', 'icts Runtime', 'cbsh-c Runtime', 'lazycbs Runtime', 'epea Runtime']

    feature_columns = ['GridColumns', 'GridRows', 'GridSize', 'NumOfObstacles', 'ObstacleDensity',
       'AvgDistanceToGoal', 'MaxDistanceToGoal', 'MinDistanceToGoal',
       'AvgStartDistances', 'AvgGoalDistances', 'PointsAtSPRatio', 'Sparsity',
       '2waycollisions', 'StdDistanceToGoal', 'MaxStartDistances',
       'MaxGoalDistances', 'MinStartDistances', 'MinGoalDistances',
       'StdStartDistances', 'StdGoalDistances']

    accuracy = 0
    coverage = 0
    coverage_runtime = 0

    accs = []
    covs = []
    runs = []
    for i, (train_df, test_df, name) in enumerate(get_split(df, split_type, test_set_number)):
        clf = XGBClassifier(objective='multi:softmax', n_jobs=-1,
                            callbacks=[WandbCallback(log_model=True,
                                                     log_feature_importance=True)])

        run.tags  = run.tags + (name[0], )

        train_df.Y = train_df.Y.apply(lambda alg: alg2label[alg])
        test_df.Y = test_df.Y.apply(lambda alg: alg2label[alg])
        clf.fit(train_df[feature_columns], train_df.Y)

        preds = torch.from_numpy(clf.predict_proba(test_df[feature_columns]))

        accuracy = calc_fastest(preds, torch.tensor(test_df.Y.tolist()))
        coverage = calc_coverage(preds, torch.from_numpy(test_df.loc[:, success_order].values.astype(np.int64)))
        coverage_runtime = calc_coverage_runtime(preds, torch.from_numpy(test_df.loc[:, runtime_order].values.astype(np.int64)))
        
        accs.append(accuracy)
        covs.append(coverage)
        runs.append(coverage_runtime)

    run.log({
        "eval/accuracy": accuracy,
        "eval/coverage": coverage,
        "eval/coverage runtime": coverage_runtime,
    })


    run.summary['accuracy'] = np.array(accs)
    run.summary['coverage'] = np.array(covs)
    run.summary['coverage_runtime'] = np.array(runs)
        

def train_ensemble(df, split_type, images_path, hparams, test_set_number):
    run = wandb.init(
    # Set the project where this run will be logged
    project="MAPF-ViT",
    # Track hyperparameters and run metadata
    tags=['Ensemble'])

    success_order = ['sat Success', 'icts Success', 'cbsh-c Success', 'lazycbs Success', 'epea Success']
    runtime_order = ['sat Runtime', 'icts Runtime', 'cbsh-c Runtime', 'lazycbs Runtime', 'epea Runtime']

    feature_columns = ['GridColumns', 'GridRows', 'GridSize', 'NumOfObstacles', 'ObstacleDensity',
       'AvgDistanceToGoal', 'MaxDistanceToGoal', 'MinDistanceToGoal',
       'AvgStartDistances', 'AvgGoalDistances', 'PointsAtSPRatio', 'Sparsity',
       '2waycollisions', 'StdDistanceToGoal', 'MaxStartDistances',
       'MaxGoalDistances', 'MinStartDistances', 'MinGoalDistances',
       'StdStartDistances', 'StdGoalDistances']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transforms = transforms.Compose(
    [
        transforms.Resize((hparams["image_size"], hparams["image_size"]), antialias=True),
        transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ]
    )

    validation_transforms = transforms.Compose(
        [
            transforms.Resize((hparams["image_size"], hparams["image_size"]), antialias=True),
            transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ]
    )

    for i, (train_df, test_df, name) in enumerate(get_split(df, split_type, test_set_number)):

        # Train XGBoost model

        clf = XGBClassifier(objective='multi:softmax', n_jobs=-1,
                            callbacks=[WandbCallback(log_model=True,
                                                     log_feature_importance=True)])

        run.tags  = run.tags + (name[0], )

        train_df.Y = train_df.Y.apply(lambda alg: alg2label[alg])
        test_df.Y = test_df.Y.apply(lambda alg: alg2label[alg])
        clf.fit(train_df[feature_columns], train_df.Y)

        preds = torch.from_numpy(clf.predict_proba(test_df[feature_columns]))

        # Train ViT model

        train_data = MAPFDataset(images_path, train_df, transform=train_transforms)
        val_data = MAPFDataset(images_path, test_df, transform=validation_transforms)

        train_loader = DataLoader(dataset = train_data, batch_size=hparams["batch_size"], shuffle=True, num_workers=8)
        valid_loader = DataLoader(dataset = val_data, batch_size=hparams["batch_size"], shuffle=False, num_workers=8)

        run.tags  = run.tags + (name[0], images_path.split('/')[-1])
        print(f'train data len: {len(train_data)}')
        print(f'test data len: {len(val_data)}')
        print(f'train loader len: {len(train_loader)}')
        print(f'test loader len: {len(valid_loader)}')

        model = SimpleViT(
            image_size = hparams["image_size"],
            patch_size = hparams["patch_size"],
            num_classes = len(alg2label.keys()),
            dim = hparams["dim"],
            depth = hparams["depth"],
            heads = hparams["heads"],
            mlp_dim = hparams["mlp_dim"]
        ).to(device)

        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'ViT parameters: {model_params}')

        # loss function
        criterion = nn.CrossEntropyLoss()
        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=hparams["lr"])
        # scheduler
        # scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

        for epoch in range(hparams["epochs"]):
            train_metrics = train_one_epoch(model, criterion, optimizer, train_loader, device)

            validation_metrics = ensemble_validation_step(model, criterion, valid_loader, device)
            

            print(f"{train_metrics=}")
            print(f"{validation_metrics=}")

            run.log(train_metrics, commit=False)
            run.log(validation_metrics, commit=True)

        
        accuracy = calc_fastest(preds, torch.tensor(test_df.Y.tolist()))
        coverage = calc_coverage(preds, torch.from_numpy(test_df.loc[:, success_order].values.astype(np.int64)))
        coverage_runtime = calc_coverage_runtime(preds, torch.from_numpy(test_df.loc[:, runtime_order].values.astype(np.int64)))
        
        accs.append(accuracy)
        covs.append(coverage)
        runs.append(coverage_runtime)

    run.log({
        "eval/accuracy": accuracy,
        "eval/coverage": coverage,
        "eval/coverage runtime": coverage_runtime,
    })


    run.summary['accuracy'] = np.array(accs)
    run.summary['coverage'] = np.array(covs)
    run.summary['coverage_runtime'] = np.array(runs)

def train_vit(df, split_type, images_path, hparams, test_set_number=None):
    run = wandb.init(
    # Set the project where this run will be logged
    project="MAPF-ViT",
    # Track hyperparameters and run metadata
    tags=['ViT'])


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transforms = transforms.Compose(
    [
        transforms.Resize((hparams["image_size"], hparams["image_size"]), antialias=True),
        transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ]
    )

    validation_transforms = transforms.Compose(
        [
            transforms.Resize((hparams["image_size"], hparams["image_size"]), antialias=True),
            transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ]
    )

    for i, (train_df, test_df, name) in enumerate(get_split(df, split_type, test_set_number)):
        train_data = MAPFDataset(images_path, train_df, transform=train_transforms)
        val_data = MAPFDataset(images_path, test_df, transform=validation_transforms)

        train_loader = DataLoader(dataset = train_data, batch_size=hparams["batch_size"], shuffle=True, num_workers=8)
        valid_loader = DataLoader(dataset = val_data, batch_size=hparams["batch_size"], shuffle=False, num_workers=8)

        run.tags  = run.tags + (name[0], images_path.split('/')[-1])
        print(f'train data len: {len(train_data)}')
        print(f'test data len: {len(val_data)}')
        print(f'train loader len: {len(train_loader)}')
        print(f'test loader len: {len(valid_loader)}')

        model = SimpleViT(
            image_size = hparams["image_size"],
            patch_size = hparams["patch_size"],
            num_classes = len(alg2label.keys()),
            dim = hparams["dim"],
            depth = hparams["depth"],
            heads = hparams["heads"],
            mlp_dim = hparams["mlp_dim"]
        ).to(device)

        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'ViT parameters: {model_params}')

        # loss function
        criterion = nn.CrossEntropyLoss()
        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=hparams["lr"])
        # scheduler
        # scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

        for epoch in range(hparams["epochs"]):
            train_metrics = train_one_epoch(model, criterion, optimizer, train_loader, device)

            validation_metrics = validation_step(model, criterion, valid_loader, device)
            

            print(f"{train_metrics=}")
            print(f"{validation_metrics=}")

            run.log(train_metrics, commit=False)
            run.log(validation_metrics, commit=True)

def train_vivit(df, split_type, images_path, hparams, test_set_number=None):
    run = wandb.init(
    # Set the project where this run will be logged
    project="MAPF-ViT",
    # Track hyperparameters and run metadata
    tags=['ViViT'])


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transforms = transforms.Compose(
    [
        transforms.Resize((hparams["image_size"], hparams["image_size"]), antialias=True),
        transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ]
    )

    validation_transforms = transforms.Compose(
        [
            transforms.Resize((hparams["image_size"], hparams["image_size"]), antialias=True),
            transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ]
    )

    for i, (train_df, test_df, name) in enumerate(get_split(df, split_type, test_set_number)):
        train_data = ViViTMAPFDataset(images_path, train_df, transform=train_transforms)
        val_data = ViViTMAPFDataset(images_path, test_df, transform=validation_transforms)

        train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
        valid_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False)

        run.tags  = run.tags + (name[0], images_path.split('/')[-1])

        print(f'train data len: {len(train_data)}')
        print(f'test data len: {len(val_data)}')
        print(f'train loader len: {len(train_loader)}')
        print(f'test loader len: {len(valid_loader)}')


        model = ViVit(
            image_size = hparams["image_size"],
            image_patch_size = hparams["patch_size"],
            num_classes = len(alg2label.keys()),
            dim = hparams["dim"],
            spatial_depth = hparams["depth"],
            heads = hparams["heads"],
            mlp_dim = hparams["mlp_dim"],
            frames = 512,
            frame_patch_size = hparams["frame_patch_size"],
            temporal_depth = hparams["temporal_depth"],
        ).to(device)

        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'ViViT parameters: {model_params}')

        # loss function
        criterion = nn.CrossEntropyLoss()
        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=hparams["lr"])
        # scheduler
        # scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

        for epoch in range(hparams["epochs"]):
            train_metrics = train_one_epoch(model, criterion, optimizer, train_loader, device, accumulation_steps=hparams["batch_size"])

            validation_metrics = validation_step(model, criterion, valid_loader, device)

            print(f"{train_metrics=}")
            print(f"{validation_metrics=}")

            run.log(train_metrics, commit=False)
            run.log(validation_metrics, commit=True)


def train_one_epoch(model, criterion, optimizer, train_loader, device, accumulation_steps=1):
    epoch_loss = 0
    epoch_accuracy = 0
    epoch_coverage = 0
    epoch_coverage_runtime = 0

    model.train()

    for idx, (data, labels, successes, runtimes) in tqdm(enumerate(train_loader)):
        data = data.to(device)
        labels = labels.to(device)
        successes = successes.to(device)
        runtimes = runtimes.to(device)

        output = model(data)
        loss = criterion(output, labels)

        # Normalize the Gradients
        loss = loss / accumulation_steps

        # scaler.scale(loss).backward()
        loss.backward()

        if ((idx + 1) % accumulation_steps == 0) or (idx + 1 == len(train_loader)):

            # Unscales the gradients of optimizer's assigned params in-place
            # scaler.unscale_(optimizer)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            # scaler.step(optimizer)
            # scaler.update()
            optimizer.step()
            optimizer.zero_grad()


        epoch_loss += loss / len(train_loader) 

        fastest_accuracy = calc_fastest(output, labels)
        epoch_accuracy += fastest_accuracy / len(train_loader)

        coverage_accuracy = calc_coverage(output, successes)
        epoch_coverage += coverage_accuracy / len(train_loader)

        coverage_runtime_accuracy = calc_coverage_runtime(output, runtimes)
        epoch_coverage_runtime += coverage_runtime_accuracy / len(train_loader)

    return {
        "train/accuracy": epoch_accuracy,
        "train/coverage": epoch_coverage,
        "train/coverage runtime": epoch_coverage_runtime,
        "train/loss": epoch_loss,
    }


def validation_step(model, criterion, valid_loader, device):
    model.eval()
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_coverage = 0
        epoch_val_coverage_runtime = 0
        epoch_val_loss = 0


        for data, labels, successes, runtimes in tqdm(valid_loader, leave=False):
            data = data.to(device)
            labels = labels.to(device)
            successes = successes.to(device)
            runtimes = runtimes.to(device)
            
            with amp.autocast():
                val_output = model(data)
                val_loss = criterion(val_output, labels)

            epoch_val_loss += val_loss / len(valid_loader) 

            fastest_accuracy = calc_fastest(val_output, labels)
            epoch_val_accuracy += fastest_accuracy / len(valid_loader)

            coverage_accuracy = calc_coverage(val_output, successes)
            epoch_val_coverage += coverage_accuracy / len(valid_loader)

            coverage_accuracy_runtime = calc_coverage_runtime(val_output, runtimes)
            epoch_val_coverage_runtime += coverage_accuracy_runtime / len(valid_loader)
    
    return {
        "eval/accuracy": epoch_val_accuracy,
        "eval/coverage": epoch_val_coverage,
        "eval/coverage runtime": epoch_val_coverage_runtime,
        "eval/loss": epoch_val_loss
    }

def ensemble_validation_step(model, criterion, valid_loader, device, xgboost_preds):
    model.eval()
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_coverage = 0
        epoch_val_coverage_runtime = 0
        epoch_val_loss = 0


        preds = []
        labels = []
        successes = []
        runtimes = []
        for data, label, success, runtime in tqdm(valid_loader, leave=False):
            data = data.to(device)
            label = label.to(device)
            success = success.to(device)
            runtime = runtime.to(device)
            
            with amp.autocast():
                val_output = model(data)
                val_loss = criterion(val_output, labels)

            epoch_val_loss += val_loss / len(valid_loader) 
            
            preds.append(val_output.cpu().numpy().tolist())
            labels.append(label.cpu().numpy().tolist())
            successes.append(success.cpu().numpy().tolist())
            runtimes.append(runtime.cpu().numpy().tolist())

        emsemble_preds = (xgboost_preds + torch.tensor(preds)) / 2
        labels = torch.tensor(labels)
        successes = torch.tensor(successes)
        runtimes = torch.tensor(runtimes)

        fastest_accuracy = calc_fastest(ensemble_preds, labels)
        epoch_val_accuracy = fastest_accuracy

        coverage_accuracy = calc_coverage(ensemble_preds, successes)
        epoch_val_coverage = coverage_accuracy

        coverage_accuracy_runtime = calc_coverage_runtime(ensemble_preds, runtimes)
        epoch_val_coverage_runtime = coverage_accuracy_runtime
    
    return {
        "eval/accuracy": epoch_val_accuracy,
        "eval/coverage": epoch_val_coverage,
        "eval/coverage runtime": epoch_val_coverage_runtime,
        "eval/loss": epoch_val_loss
    }
