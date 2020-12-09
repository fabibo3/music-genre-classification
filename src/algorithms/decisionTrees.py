"""
Decision Tree algorithm
"""
__author__ = "Fabian Bongratz"

import numpy as np
import torch
from datasets.datasets import MusicDataset
from utils.utils import accuracy
from catboost import CatBoostClassifier, Pool, cv
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
    params['eval_metric'] = 'Accuracy'
    params['loss_function'] = config.get('loss_function', 'CrossEntropy')
    params['iterations'] = config.get(_n_iterations_key, 10)
    params['learning_rate'] = config.get(_learning_rate_key, 0.1)

    train_split = 0.9
    early_stop = config.get('early_stop', False)


    # Get data
    _, X, y = train_dataset.get_whole_dataset_as_pd()
    if(early_stop):
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8,
                                                          random_state=0)
        eval_set = (X_val, y_val)
    else:
        eval_set = None

    test_files, X_test, _ = test_dataset.get_whole_dataset_as_pd()

    # GPU
    if(torch.cuda.is_available()):
        task_type = 'GPU'
        devices = str(torch.cuda.get_current_device())
    else:
        task_type = 'CPU'
        devices = None
    params['task_type'] = task_type
    params['devices'] = devices

    model = CatBoostClassifier(**params)
    model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=50,
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
                               val_dataset: MusicDataset=None,
                               internal_cv=False):
    """
    Fit a CatBoostClassifier using train and validation set
    Returns:
        - a list of the names of the parameters
        - a list of tried parameter configurations
        - a list of corresponding results
    """
    # Get parameters
    if(type(config[_n_iterations_key])==list):
        iterations = np.arange(config[_n_iterations_key][0],
                                config[_n_iterations_key][1],
                                config[_n_iterations_key][2])
    else:
        iterations = config[_n_iterations_key]

    if(type(config[_learning_rate_key])==list):
        learning_rates = np.arange(config[_learning_rate_key][0],
                                config[_learning_rate_key][1],
                                config[_learning_rate_key][2])
    else:
        learning_rates = config[_learning_rate_key]

    loss_function = config.get("loss_function", "CrossEntropy")
    parameter_names = []
    parameter_sets = []
    results = []

    # Get data
    _, X_train, y_train = train_dataset.get_whole_dataset_as_pd()
    if(val_dataset != None):
        _, X_val, y_val = val_dataset.get_whole_dataset_as_pd()

    # GPU
    if(torch.cuda.is_available()):
        task_type = 'GPU'
        devices = str(torch.cuda.current_device())
    else:
        task_type = 'CPU'
        devices = None

    if(not internal_cv):
        # No internal cross validation during training
        for i_it, it in enumerate(iterations):
            for i_lr, lr in enumerate(learning_rates):
                model = CatBoostClassifier(
                        iterations=it,
                        learning_rate=lr,
                        loss_function=loss_function,
                        task_type=task_type,
                        devices=devices,
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
    else:
        # Use catboost cross validation procedure
        params = {}
        params['loss_function'] = loss_function
        params['iterations'] = iterations
        params['custom_metric'] = 'Accuracy'
        params['task_type'] = task_type
        params['devices'] = devices

        best_value = 0.0
        best_iter = 0
        for i_lr, lr in enumerate(learning_rates):
            params['learning_rate'] = lr
            cv_data = cv(
                    params = params,
                    pool = Pool(X_train, label=y_train),
                    fold_count=5,
                    shuffle=True,
                    partition_random_seed=0,
                    plot=True,
                    stratified=False,
                    verbose=50
            )
            res_value = np.max(cv_data['test-Accuracy-mean'])
            res_iter = np.argmax(cv_data['test-Accuracy-mean'])
            params['best_iteration'] = res_iter

            print(f"Best iteration for lr {lr}: {res_iter} with val accuracy {res_value}")

            results.append(res_value)
            parameter_sets.append(list(params.values()))
            parameter_names = list(params.keys())

            # Remove entry from dict since it is used as input for cv again
            params.pop('best_iteration')

    return parameter_names, parameter_sets, results

