"""
k-Nearest-Neighbors algorihm
"""
__author__ = "Fabian Bongratz"

import numpy as np
from datasets.datasets import MusicDataset
from utils.utils import precision
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

_n_iterations_key = "iterations"
_learning_rate_key = "learning_rate"

def run_decisionTree(config: dict,
                     train_dataset: MusicDataset,
                     test_dataset: MusicDataset) -> dict:
    """
    Fit a decisionTree model on the training set and run it on the test set
    afterwards
    @param config: A configuration defining the algorithm parameters
    @param train_dataset: The training data as MusicDataset
    @param test_dataset: The test data as MusicDataset
    ------
    @return predictions for the test data
    """
    params = {}
    params['custom_metric'] = 'Accuracy'
    params['loss_function'] = config.get('loss_function', 'CrossEntropy')
    params['iterations'] = config.get(_n_iterations_key, 10)
    params['learning_rate'] = config.get(_learning_rate_key, 0.1)

    train_split = 0.9
    early_stop = config.get('early_stop', False)


    # Get data
    _, X_train, y_train = train_dataset.get_whole_dataset_as_pd()
    test_files, X_test, _ = test_dataset.get_whole_dataset_as_pd()

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      train_size=train_split, random_state=1234)

    model = CatBoostClassifier(**params)
    model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=10,
            plot=True
    )
    result = model.predict(X_test, prediction_type='Class',
                                verbose=10).flatten()
    predictions = {}
    for i, file_id in enumerate(test_files):
        predictions[file_id] = result[i]

    return predictions

def search_CatBoost_parameters(config: dict,
                   train_dataset: MusicDataset,
                   val_dataset: MusicDataset):
    """
    Fit a CatBoostClassifier using train and validation set
    Returns:
        - a list of the names of the parameters
        - a list of tried parameter configurations
        - a list of corresponding results
    """
    # Get parameters
    iterations = np.arange(config[_n_iterations_key][0],
                            config[_n_iterations_key][1],
                            config[_n_iterations_key][2])
    learning_rates = np.arange(config[_learning_rate_key][0],
                            config[_learning_rate_key][1],
                            config[_learning_rate_key][2])
    loss_function = config.get("loss_function", "CrossEntropy")
    parameter_names = []
    parameter_sets = []
    results = []

    # Get data
    _, X_train, y_train = train_dataset.get_whole_dataset_as_pd()
    _, X_val, y_val = val_dataset.get_whole_dataset_as_pd()

    for i_it, it in enumerate(iterations):
        for i_lr, lr in enumerate(learning_rates):
            model = CatBoostClassifier(
                    iterations=it,
                    learning_rate=lr,
                    loss_function=loss_function,
                    custom_metric=['Accuracy']
            )
            model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    verbose=10
            )
            params = model.get_params()
            parameter_names = list(params.keys())
            parameter_sets.append(list(params.values()))
            best_score = model.get_best_score()
            results.append(best_score['validation']['Accuracy'])
            best_iter = model.get_best_iteration()
            print("Best iteration: " + str(best_iter))

    return parameter_names, parameter_sets, results

