"""
Procedures for preparation of experiments
"""
__author__ = "Fabian Bongratz"

import os
import random
from shutil import copyfile
from utils.config import load_config_file
from utils.folders import get_experiment_folder,\
                            get_config_file_path,\
                            get_train_data_path
from algorithms.kNN import run_kNN, search_kNN_parameters
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

    # Distinguish between test, train and parameter-search
    if(config["mode"]=="train"):
        raise NotImplementedError
    elif(config["mode"]=="parameter-search"):
        search_parameters(config)
    else:
        run_test(config)

    print("#"*50)
    print(f"Experiment {exp_name} finished.")

def search_parameters(config: str):
    """
    Find good hyperparameters by evaluation on validation split
    """
    print("#"*50)
    # Define datasets
    dataset_config = config["dataset"]
    train_split = dataset_config["train_split"]
    val_split = dataset_config["val_split"]
    all_files = os.listdir(get_train_data_path())
    random.shuffle(all_files)
    n_train = int(train_split/100.0 * len(all_files))
    n_val = int(val_split/100.0 * len(all_files))
    train_dataset = MusicDataset(split="train",
                                 mfcc_file="mfccs.csv",
                                 files=all_files[:n_train])
    print(f"Using {len(train_dataset)} training files")
    val_dataset = MusicDataset(split="train",
                               mfcc_file="mfccs.csv",
                               files=all_files[-n_val:])
    print(f"Using {len(val_dataset)} validation files")
    print("Datasets created")

    # Algorithm configuration
    algo_config = config[_algorithm_name_key]

    if(algo_config["type"]=="kNN"):
        parameters = search_kNN_parameters(algo_config, train_dataset, val_dataset)
    else:
        raise NotImplementedError("Algorithm not implemented!")

    # Write parameters to csv
    exp_name = config[_experiment_name_key]
    parameter_file = os.path.join(get_experiment_folder(exp_name), "tuned_parameters.csv")
    write_parameters_to_csv(parameter_file, parameters)
    print(f"Best model written to {parameter_file}")

def write_parameters_to_csv(parameter_file, parameters):
    with open(parameter_file, 'w') as f:
        f.write("parameter,value\n")
        for p,v in parameters.items():
            f.write(f"{p},{v}\n")



def run_test(config: str):
    """
    Evaluate a certain model on the test set
    """
    # Define datasets
    test_dataset = MusicDataset(split="test", mfcc_file="mfccs.csv")
    print("#"*50)
    print("Datasets created")

    # Algorithm configuration
    algo_config = config[_algorithm_name_key]

    # Run algorithm defined by the config
    predictions = run_algorithm(algo_config, test_dataset)

    # Write result to csv
    exp_name = config[_experiment_name_key]
    result_file = os.path.join(get_experiment_folder(exp_name), "predictions.csv")
    write_result_to_csv(result_file, predictions)

def write_result_to_csv(result_file: str, predictions: dict):
    with open(result_file, 'w') as f:
        f.write("track_id,genre_id\n")
        for k,v in predictions.items():
            f.write(f"{k.split('.')[0]},{str(v)}\n")

def run_algorithm(algo_config: str, dataset: MusicDataset):
    """
    Run an algorithm defined by algo_config on a certain dataset
    """
    print("#"*50)
    print("Start algorithm")

    if(algo_config["type"]=="kNN"):
        train_dataset = MusicDataset(split="train", mfcc_file="mfccs.csv")
        predictions = run_kNN(algo_config, train_dataset, dataset)
    else:
        raise ValueError("Algorithm not known!")

    # Predict random class for non-valid test data
    all_ids = dataset.get_all_files()
    for i in all_ids:
        if(i not in predictions):
            predictions[i] = 1

    print("Algorithm finished")

    return predictions

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

