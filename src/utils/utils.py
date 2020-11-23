"""
Utility functions
"""
__author__ = "Fabian Bongratz"

import numpy as np

def precision(prediction, ground_truth):
    return float(np.sum(prediction==ground_truth))/len(ground_truth)
