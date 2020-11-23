"""
k-Nearest-Neighbors algorihm
"""
__author__ = "Fabian Bongratz"

import numpy as np
from sklearn import neighbors
from datasets.datasets import MusicDataset
from utils.utils import precision

_n_neighbors_key = "n_neighbors"
_n_mfcc_coeffs_key = "n_mfcc_coefficients"

def run_kNN(config: dict,
            train_dataset: MusicDataset,
            test_dataset:
            MusicDataset,
            n_features=-1) -> dict:
    """
    Run a k-NN classifier defined by several parameters given a train and a
    test set
    """
    # Process config
    if("weights" in config):
        weights = config["weights"]
    else:
        weights = "uniform"
    if("n_neighbors" in config):
        n_neighbors = config["n_neighbors"]
    else:
        n_neighbors = 1

    # Get data
    train_files, train_mfccs, train_labels = train_dataset.get_whole_dataset()
    test_files, test_mfccs, _ = test_dataset.get_whole_dataset()

    # Create k-NN classifiers
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    train_labels = np.asarray(train_labels)

    # Use only a subset of the available features
    if(n_features > 0):
        if(n_features <= train_mfccs.shape[1]):
            train_mfccs = train_mfccs[:,:n_features]
            test_mfccs = test_mfccs[:,:n_features]
        else:
            raise ValueError("Not enough features available")

    clf.fit(train_mfccs, train_labels)
 
    #Predict
    result = clf.predict(test_mfccs)
    predictions = {}
    for i, file_id in enumerate(test_files):
        predictions[file_id] = result[i]

    return predictions

def search_kNN_parameters(config: dict,
                          train_dataset: MusicDataset,
                          val_dataset: MusicDataset):
    """
    Perform grid-hyperparameter search for a kNN classifier
    """
    print("#"*50)
    print("Searching for best parameters...")
    n_neighbors = np.arange(config[_n_neighbors_key][0],
                            config[_n_neighbors_key][1],
                            config[_n_neighbors_key][2])
    n_mfcc_coeffs = np.arange(config[_n_mfcc_coeffs_key][0],
                            config[_n_mfcc_coeffs_key][1],
                            config[_n_mfcc_coeffs_key][2])
    kNN_config=config.copy()
    result_shape = (len(n_neighbors), len(n_mfcc_coeffs))
    results = np.zeros(result_shape)

    for i_k,k in enumerate(n_neighbors):
        for i_n_coeffs,n_coeffs in enumerate(n_mfcc_coeffs):
            kNN_config[_n_neighbors_key] = k
            kNN_config[_n_mfcc_coeffs_key] = n_coeffs

            predictions = run_kNN(kNN_config,
                                  train_dataset,
                                  val_dataset,
                                  n_features=n_coeffs)
            _, _, ground_truth = val_dataset.get_whole_dataset()
            predictions = list(predictions.values())
            assert(len(predictions)==len(ground_truth))
            results[i_k,i_n_coeffs] = precision(np.asarray(predictions),
                                                np.asarray(ground_truth)) 

    # Get best results
    best_indices = np.unravel_index(np.argmax(results), result_shape)
    best_k = n_neighbors[best_indices[0]]
    best_n_mfcc = n_mfcc_coeffs[best_indices[1]]

    return {"n_neighbors": best_k, "n_mfcc_coefficients": best_n_mfcc}

    

