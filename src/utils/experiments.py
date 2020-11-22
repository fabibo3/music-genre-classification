"""
Procedures for preparation of experiments
"""
__author__ = "Fabian Bongratz"

import os
from shutil import copyfile
from utils.config import load_config_file
from utils.folders import get_experiment_folder, get_config_file_path
from algorithms.kNN import run_kNN

# Keys of the json config file
_experiment_name_key = "experiment_name"
_algorithm_name_key = "algorithm"

def run_experiment(config_file: str):
    """
    Run an experiment defined by a config file
    """
    config = prepare_experiment(config_file)
    algo_config = config[_algorithm_name_key]

    if(algo_config["type"]=="kNN"):
        run_kNN(algo_config)
    else:
        raise ValueError("Algorithm not known!")


def prepare_experiment(config_file: str):
    # Get json as dict
    config = load_config_file(config_file)

    if _experiment_name_key not in config:
        config[_experiment_name_key] = config_file

    exp_name = config[_experiment_name_key]
    print(f"Preparing experiment \"{exp_name}\"...")

    # Create experiment folder
    exp_folder = get_experiment_folder(exp_name)
    os.makedirs(exp_folder)

    # Copy the config file to the experiment folder
    config_file_path = get_config_file_path(config_file)
    copyfile(config_file_path, f'{exp_folder}/{exp_name}.json')

    return config

def get_experiment_name(config: dict) -> str:
    return config[_experiment_name_key]

