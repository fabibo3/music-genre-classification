"""
Project folder structure
"""
__author__ = "Fabian Bongratz"

from os import path

# Root path of this project, computed based on absolute path of this file

root_folder = path.abspath(path.join(path.dirname(path.relpath(__file__)),
                                     path.pardir, path.pardir))

config_folder = f'{root_folder}/config'
data_base_folder = f'{root_folder}/data'
experiments_folder = f'{root_folder}/experiments'

def set_dataset_base_folder(folder: str):
    global data_base_folder 
    data_base_folder = folder

def get_experiment_folder(experiment_name: str) -> str:
    return f'{experiments_folder}/{experiment_name}'

def get_dataset_base_folder() -> str:
    return data_base_folder

def get_config_file_path(config_name: str) -> str:
    return f'{config_folder}/{config_name}'

def get_train_data_path() -> str:
    """
    Get the path of all training mp3 files
    """
    return f'{data_base_folder}/train/Train'

def get_test_data_path() -> str:
    """
    Get the path of all test mp3 files
    """
    return f'{data_base_folder}/test/Test'

def get_preprocessed_data_path(split: str) -> str:
    """
    Get either train or test set path of preprocessed features
    @param split: "train" or "test"
    """
    assert(split=="test" or split=="train")
    return f'{data_base_folder}/{split}/preprocessed'
