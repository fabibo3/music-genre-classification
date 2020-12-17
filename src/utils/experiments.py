"""
Procedures for preparation of experiments
"""
__author__ = "Fabian Bongratz"

import os
import random
import numpy as np
from shutil import copyfile
from utils.config import load_config_file
from utils.folders import get_experiment_folder,\
                            get_config_file_path,\
                            get_train_data_path,\
                            get_preprocessed_data_path,\
                            get_dataset_base_folder
from algorithms.kNN import run_kNN, search_kNN_parameters
from algorithms.decisionTrees import search_CatBoost_parameters,\
run_decisionTree
from algorithms.neural_networks import run_nn_model, search_nn_parameters 
from datasets.datasets import MusicDataset, MelSpectroDataset

# Keys of the json config file
_experiment_name_key = "experiment_name"
_algorithm_name_key = "algorithm"
_search_param_config_key = "parameter-search"

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
        # Train is equal to parameter search without parameter ranges
        search_parameters(config)
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
    # Define datasets
    dataset_config = config["dataset"]
    train_split = dataset_config["train_split"]
    val_split = dataset_config.get("val_split", 0)
    dataset_type = dataset_config.get("features", "mp3") #mp3 MusicDataset by default
    dataset_shuffle = dataset_config.get("shuffle", False) 

    search_param_config = config[_search_param_config_key]

    # Cross validation on/off
    cross_val_on = search_param_config.get("exterior_cross_validation", False)
    if(cross_val_on and val_split > 0):
        n_runs = int(100/val_split)
    else:
        n_runs = 1


    parameter_names = []
    parameter_sets = []
    results = []
 
    # Prepare datasets
    if(dataset_type=="melspectro"):
        data_path = os.path.join(get_dataset_base_folder(),
                                  "melspectro_songs_train_new.pickle")
        label_path = os.path.join(get_dataset_base_folder(),
                     "melspectro_genres_train_new.pickle")
        dataset = MelSpectroDataset(data_path, label_file=label_path)
        n_train = int(train_split/100.0 * len(dataset))
        n_val = int(val_split/100.0 * len(dataset))
        # Shuffle
        if(dataset_shuffle):
            data_indices = np.random.permutation(len(dataset))
        else:
            data_indices = np.arange(len(dataset))
    elif(dataset_type=="vgg_features"):
        data_path = os.path.join(get_preprocessed_data_path("train"),
                                  "vgg_train.pickle")
        label_path = os.path.join(get_dataset_base_folder(), "train.csv")
        file_names_path = os.path.join(get_preprocessed_data_path("train"),
                                       "valid_ids_sorted.pickle")
        dataset = MelSpectroDataset(data_path, label_file=label_path,
                                    file_names_file=file_names_path)
        n_train = int(train_split/100.0 * len(dataset))
        n_val = int(val_split/100.0 * len(dataset))
        # Shuffle
        if(dataset_shuffle):
            data_indices = np.random.permutation(len(dataset))
        else:
            data_indices = np.arange(len(dataset))
    else:
        n_train = int(train_split/100.0 * len(all_files))
        n_val = int(val_split/100.0 * len(all_files))
        all_files = os.listdir(get_train_data_path())
        # Shuffle
        if(dataset_shuffle):
            data_indices = np.random.permutation(len(all_files))
        else:
            data_indices = np.arange(len(all_files))

    data_indices = data_indices.tolist()

    print("#"*50)
    print("Searching for best parameters...")

    for i in range(n_runs):
        if(dataset_type == "melspectro" or dataset_type == "vgg_features"):
            # Split into train/validation
            if(dataset_type == "melspectro"):
                train_dataset = MelSpectroDataset(data_path, label_file=label_path)
            if(dataset_type == "vgg_features"):
                train_dataset = MelSpectroDataset(data_path, label_file=label_path,
                                    file_names_file=file_names_path)
            train_dataset.set_subset(data_indices[:n_train])

            print(f"Using {len(train_dataset)} training files")
            if(val_split > 0):
                val_dataset = MelSpectroDataset(data_path, label_file=label_path,
                                    file_names_file=file_names_path)
                val_dataset.set_subset(data_indices[-n_val:])
                print(f"Using {len(val_dataset)} validation files")
        else:
            # Split into train/validation
            train_dataset = MusicDataset(split="train",
                                         mfcc_file="mfccs.csv",
                                         files=all_files[data_indices[:n_train]])
            print(f"Using {len(train_dataset)} training files")
            if(val_split > 0):
                val_dataset = MusicDataset(split="train",
                                           mfcc_file="mfccs.csv",
                                           files=all_files[data_indices[-n_val:]])
                print(f"Using {len(val_dataset)} validation files")
            else:
                val_dataset = None

        print("Datasets created")

        # Algorithm configuration
        algo = config[_algorithm_name_key]
        algo_config = config[algo]

        if(algo =="kNN"):
            parameter_names, \
                    parameter_sets,\
                    cur_results = search_kNN_parameters(algo_config,
                                                        train_dataset,
                                                        val_dataset)
        elif(algo == "decision-tree"):
            internal_cross_val_on = search_param_config.get("internal_cross_validation", False)
            parameter_names,\
                    parameter_sets,\
                    cur_results = search_CatBoost_parameters(algo_config,
                                                             train_dataset,
                                                             val_dataset,
                                                             internal_cv=internal_cross_val_on)

        elif(algo == "neural-network"):
            parameter_names,\
                    parameter_sets,\
                    cur_results = search_nn_parameters(algo_config,
                                                       config[_experiment_name_key], train_dataset, val_dataset)
        else:
            raise NotImplementedError("Algorithm not implemented!")

        assert(len(parameter_names) == len(parameter_sets[0]))

        results.append(cur_results)

        # Rotate files/data samples to get different splits
        data_indices = data_indices[-n_val:] + data_indices[:n_train]


    # Get the best configuration
    results = np.median(np.asarray(results), axis=0)
    max_res = np.max(results)
    imax = np.argmax(results)

    print("#"*50)
    print(f"Best value: {max_res}")
    print("Best parameters found:")

    # Extract corresponding parameters
    parameters = {}
    for i,pn in enumerate(parameter_names):
        param_choice = parameter_sets[imax][i]
        print(f"{pn}: {param_choice}")
        parameters.update({pn: param_choice})

    # Write parameters to csv
    exp_name = config[_experiment_name_key]
    parameter_file = os.path.join(get_experiment_folder(exp_name), "tuned_parameters.csv")
    write_parameters_to_csv(parameter_file, parameters)
    print(f"Best parameters of search written to {parameter_file}")

def write_parameters_to_csv(parameter_file, parameters):
    with open(parameter_file, 'w') as f:
        f.write("parameter,value\n")
        for p,v in parameters.items():
            f.write(f"{p},{v}\n")



def run_test(config: str):
    """
    Evaluate a certain model on the test set
    """
    dataset_type = config['dataset']['features']
    # Define datasets
    if(dataset_type=="melspectro"):
        data_path = os.path.join(get_dataset_base_folder(),
                                  "melspectro_songs_test_new.pickle")
        file_names_file = os.path.join(get_dataset_base_folder(),
                     "melspectro_filenames_test.pickle")
        test_dataset = MelSpectroDataset(data_path, file_names_file=file_names_file)
    elif(dataset_type=="vgg_features"):
        data_path = os.path.join(get_preprocessed_data_path("test"),
                                  "vgg_test.pickle")
        file_names_file = os.path.join(get_dataset_base_folder(),
                     "test.csv")
        test_dataset = MelSpectroDataset(data_path, file_names_file=file_names_file)
    else:
        test_dataset = MusicDataset(split="test", mfcc_file="mfccs.csv")
    print("#"*50)
    print("Datasets created")

    # Algorithm configuration
    algo = config[_algorithm_name_key]
    algo_config = config[algo]

    # Run algorithm defined by the config
    predictions = run_algorithm(algo_config, test_dataset,
                                config[_experiment_name_key])

    # Write result to csv
    exp_name = config[_experiment_name_key]
    result_file = os.path.join(get_experiment_folder(exp_name), "predictions.csv")
    write_result_to_csv(result_file, predictions)

def write_result_to_csv(result_file: str, predictions: dict):
    with open(result_file, 'w') as f:
        f.write("track_id,genre_id\n")
        for k,v in predictions.items():
            f.write(f"{k.split('.')[0]},{str(v)}\n")

def run_algorithm(algo_config: str, dataset: MusicDataset, experiment_name):
    """
    Run an algorithm defined by algo_config on a certain dataset
    """
    print("#"*50)
    print("Start algorithm")

    if(algo_config["type"]=="kNN"):
        train_dataset = MusicDataset(split="train", mfcc_file="mfccs.csv")
        predictions = run_kNN(algo_config, train_dataset, dataset)
    elif(algo_config["type"]=="decision-tree"):
        train_dataset = MusicDataset(split="train", mfcc_file="mfccs.csv")
        predictions = run_decisionTree(algo_config, train_dataset, dataset)
    elif(algo_config["type"]=="neural-network"):
        predictions = run_nn_model(algo_config['model_path'], dataset, experiment_name)

    else:
        raise ValueError("Algorithm not known!")

    # Predict random class for non-valid test data
    entire_dataset = MusicDataset(split="test")
    all_ids = entire_dataset.get_all_files()
    for i in all_ids:
        if(i not in predictions):
            print(f"[Warning]: No predicted value for {i}")
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

