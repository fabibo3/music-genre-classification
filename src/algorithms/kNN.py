"""
k-Nearest-Neighbors algorihm
"""
__author__ = "Fabian Bongratz"

import numpy as np
from sklearn import neighbors
from datasets.datasets import MusicDataset

def run_kNN(config: dict, train_dataset: MusicDataset, test_dataset: MusicDataset) -> dict:
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
    clf.fit(train_mfccs, train_labels)

    #Predict
    result = clf.predict(test_mfccs)
    predictions = {}
    for i, file_id in enumerate(test_files):
        predictions[file_id] = result[i]

    return predictions
