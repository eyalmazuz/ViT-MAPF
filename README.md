# ViT-MAPF

# Requirements
This repository require the following packages

1. numpy
2. matplotlib
3. networkx
4. pandas
5. vit-pytorch
6. xgboost
7. seaborn
8. tqdm
9. igraph
10. torch
11. wandb
12. torchvision
13. scikit-learn

# Data
1. To generate training data download the csv files from the following link: [download](https://drive.google.com/drive/folders/13ziLw3PJzGeNGARTrgGXzRySMQh-yOQF)
and place then in the main directory of the project (next to the src folder)

2. Download all scenario files from [here](https://movingai.com/benchmarks/mapf/index.html) by selecting the ``Download all random scenarios`` ad
``Download all even scenarios`` options and extract the contents into a new directory named ``scen-all``.

3. Download all maps from the same link as step 2 but select the ``Download all maps`` option, put all the maps in a folder named ``mapf-map`` in the main directory

4. To generate image features create a folder named ``custom_mapf_images_all_paths`` in the main directory and run ``python3 src/utils/create_image_features.py`` and to generate video features create a folder named ``data_frames_map_paths_start_goal_agg`` run ``python3 src/utils/create_video_features.py``.
   You suppose to have little over 190k files in each directory all npz format.


# Running
To run experiments open ``main.py`` and uncomment which model you want to train, either: ``train_xgb`` to train XGBoost model using hand_crafted features or ``train_vit`` to train a ViT model or ``train_vivit`` to train ViViT using videos.

to change the type of split in the experiment, change the 2nd parameter from ``map_type`` to either: ``map_type``, ``grid_name`` or ``random``
