import psutil
import time
import logging
from datetime import datetime
import random
import os
import shutil
import itertools
import pandas as pd
import numpy as np
import torch

seed = 377151
torch.manual_seed(seed)

dataset_list =   ['a2', 'a3', 'a4','a12', 'a21']

# Define the base directory
base_dir = os.path.abspath('.')
results_dir = str(os.path.join(base_dir, 'results'))

# Ensure the 'results' directory exists
os.makedirs(results_dir, exist_ok=True)

# Get the current working directory
current_working_dir = os.getcwd()

# Define the relative path to the ray_results directory
ray_results_dir = os.path.join(current_working_dir, "ray_results")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



########## Loading of Data #####################################################################
# Laden der Daten
a2_df = pd.read_csv(r'data/A2.csv', index_col=0)
a3_df = pd.read_csv(r'data/A3.csv', index_col=0)
a4_df = pd.read_csv(r'data/A4.csv', index_col=0)
a12_df = pd.read_csv(r'data/A12.csv', index_col=0)
a21_df = pd.read_csv(r'data/A21.csv', index_col=0)

a2_normalized = pd.read_csv(r'data/normalized/A2.csv', index_col=0)
a3_normalized = pd.read_csv(r'data/normalized/A3.csv', index_col=0)
a4_normalized = pd.read_csv(r'data/normalized/A4.csv', index_col=0)
a12_normalized = pd.read_csv(r'data/normalized/A12.csv', index_col=0)
a21_normalized = pd.read_csv(r'data/normalized/A21.csv', index_col=0)

joined_df = pd.concat([a2_df, a3_df, a4_df, a12_df, a21_df], axis=0)

def make_five_splits(df):
    splits = {0: [], 1: [], 2: [], 3: [], 4: []}
    for track in df['track'].unique():
        df[df['track'] == track].sample(frac=1)
        for i in range(5):
            splits[i] = (df.iloc[(i * len(df) // 5):((i + 1) * len(df) // 5)])

    return splits

a2_splits = make_five_splits(a2_df)
a3_splits = make_five_splits(a3_df)
a4_splits = make_five_splits(a4_df)
a12_splits = make_five_splits(a12_df)
a21_splits = make_five_splits(a21_df)

a2_splits_normalized = make_five_splits(a2_normalized)
a3_splits_normalized = make_five_splits(a3_normalized)
a4_splits_normalized = make_five_splits(a4_normalized)
a12_splits_normalized = make_five_splits(a12_normalized)
a21_splits_normalized = make_five_splits(a21_normalized)

########### Labels for MTL ####################################################################################

eligable_labels_for_set = {'a2': ['Track3', 'Track4'],
                           'a3': ['Track2', 'Track1'],
                           'a4': ['Track4', 'Track2', 'Track1'],
                           'a12': ['Track3', 'Track8', 'Track15', 'Track11', 'Track13', 'Track7'],
                           'a21': ['Track3', 'Track4']}


def create_labels_for_mtl(df, label):
    one_hot = pd.get_dummies(df['track'])

    result = {}
    result[label] = one_hot[eligable_labels_for_set[label]].astype(float).values.reshape(-1, len(eligable_labels_for_set[label]))

    # rename the labels - appnd the dataset label to the column name
    result[label] = pd.DataFrame(result[label]).astype(float)
    result[label] = result[label].add_prefix(label + '_')
    
    encoded_labels = df.join(result.values()) 

    #task_id_df = pd.DataFrame([label] * len(df), columns=['task_id'])   
    #encoded_labels = encoded_labels.join([task_id_df, encoded_labels])

    return encoded_labels




def get_joined_train_test_folds(split_index, normalized=False):

    #create one big fold for training - consisting of data from all datasets

    if normalized:
        split_list =   {'a2': a2_splits_normalized, 'a3': a3_splits_normalized, 'a4': a4_splits_normalized, 'a12': a12_splits_normalized, 'a21': a21_splits_normalized}
    else:
        split_list =   {'a2': a2_splits, 'a3': a3_splits, 'a4': a4_splits, 'a12': a12_splits, 'a21': a21_splits}
        
    train_set = {}
    for dataset_label in split_list.keys():
        
        dataset = split_list[dataset_label]
        indices = [0, 1, 2, 3, 4]
        # already double
        indices.remove(split_index)
        
        # it is a list not a dataframe
        train_splits = pd.concat([dataset[i] for i in indices], axis=0)
        train_splits = train_splits.reset_index(drop=True)

        # add columns for encoded labels
        train_splits = create_labels_for_mtl(train_splits, dataset_label)
        # shuffle
        train_splits = train_splits.sample(frac=1, random_state=377151)
        train_splits = train_splits.reset_index(drop=True)
        train_splits.drop('track', axis=1, inplace=True)

        train_set[dataset_label] = train_splits
    

    # wieder auseinander friemeln
    y_train_set = {}
    for label in split_list.keys():
        column_labels = [label + '_' + str(i) for i in range(0,len(eligable_labels_for_set[label]))]
        y_train_set[label] = train_set[label][column_labels]
        train_set[label].drop(column_labels, axis=1, inplace=True)

    #create one big fold for testing - consisting of data from all datasets
    test_set = {}
    for dataset_label in split_list.keys():
        
        split = split_list[dataset_label][split_index]
        split = split.reset_index(drop=True)

        test_splits = create_labels_for_mtl(split, dataset_label)
        test_splits.drop('track', axis=1, inplace=True)

        # shuffle
        test_splits = test_splits.sample(frac=1, random_state=377151)
        test_splits = test_splits.reset_index(drop=True)

        test_set[dataset_label] = test_splits


    # wieder auseinander friemeln
    y_test_set = {}
    for label in split_list.keys():
        column_labels = [label + '_' + str(i) for i in range(0,len(eligable_labels_for_set[label]))]
        y_test_set[label] = test_set[label][column_labels]

        test_set[label].drop(column_labels, axis=1, inplace=True)

    return (train_set, test_set, y_train_set, y_test_set)



# new functions added by Kevin
def clear_folder(folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if item == '.gitignore':
            continue
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)