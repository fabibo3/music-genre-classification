"""
Procedures for preparation of experiments
"""
__author__ = "Fabian Bongratz"

import os
from shutil import copyfile
from utils.config import load_config_file
from utils.folders import get_experiment_folder, get_config_file_path
from algorithms.kNN import run_kNN
from datasets.datasets import MusicDataset

# Keys of the json config file
_experiment_name_key = "experiment_name"
_algorithm_name_key = "algorithm"

def run_experiment(config_file: str):
    """
    Run an experiment defined by a config file
    """
    # Prepare experiment
    print("#"*50)
    config = prepare_experiment(config_file)
    exp_name = config[_experiment_name_key]
    # Define datasets
    train_dataset = MusicDataset(split="train", mfcc_file="mfccs.csv")
    test_dataset = MusicDataset(split="test", mfcc_file="mfccs.csv")
    print("#"*50)
    print("Datasets created")

    # Algorithm configuration
    algo_config = config[_algorithm_name_key]

    print("#"*50)
    print("Start algorithm")

    if(algo_config["type"]=="kNN"):
        predictions = run_kNN(algo_config, train_dataset, test_dataset)
    else:
        raise ValueError("Algorithm not known!")

    # Predict random class for non-valid test data
    all_ids = test_dataset.get_all_files()
    for i in all_ids:
        if(i not in predictions):
            predictions[i] = 1

    print("Algorithm finished")

    # Write result to csv
    result_file = os.path.join(get_experiment_folder(exp_name), "predictions.csv")
    write_result_to_csv(result_file, predictions)

    print("#"*50)
    print(f"Experiment {exp_name} finished.")

def write_result_to_csv(result_file: str, predictions: dict):
    with open(result_file, 'w') as f:
        f.write("track_id,genre_id\n")
        for k,v in predictions.items():
            f.write(f"{k.split('.')[0]},{str(v)}\n")

def prepare_experiment(config_file: str):
    # Get json as dict
    config = load_config_file(config_file)

    if _experiment_name_key not in config:
        config[_experiment_name_key] = config_file

    exp_name = config[_experiment_name_key]
    print(f"Preparing experiment \"{exp_name}\"...")

    # Create experiment folder
    exp_folder = get_experiment_folder(exp_name)
    os.makedirs(exp_folder, exist_ok=False)


    # Copy the config file to the experiment folder
    config_file_path = get_config_file_path(config_file)
    copyfile(config_file_path, f'{exp_folder}/{exp_name}.json')

    return config

def get_experiment_name(config: dict) -> str:
    return config[_experiment_name_key]

