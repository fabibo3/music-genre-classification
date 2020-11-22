"""
Preprocessing the data (e.g. calculating MFCC)
"""
__author__ = "Fabian Bongratz"

import os
import librosa
import numpy as np
from tqdm import tqdm
from utils.folders import get_train_data_path, get_test_data_path, get_preprocessed_data_path
import argparse

def calculate_mfcc(input_path: str, output_file: str, n_mfcc=40, n_samples=-1):
    """
    Calculate the mfcc of all files in input_path and 
    store results in output_file
    """
    files = os.listdir(input_path)
    if(n_samples==-1):
        n_samples=len(files)
    files = files[:n_samples]
    with open(output_file, 'w') as out_file: 
        out_file.write(f"File Name, MFCC Coefficients 1-{n_mfcc}\n")
        for f in tqdm(files, desc="Calculate MFCCs..."):
            file_name = os.path.join(input_path, f)
            y, sr = librosa.load(file_name)
            mfcc = librosa.feature.mfcc(y, sr, n_mfcc=n_mfcc)

            # Take mean of frames
            mfcc = np.mean(mfcc, axis=1)

            # Write to file
            line = ",".join(map(str, mfcc.tolist()))
            line = line + "\n"
            out_file.write(line)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples",
                        dest='n_samples',
                        default=-1,
                        action='store',
                        type=int,
                        nargs='?',
                        help="Restrict preprocessing to n samples")
    args = parser.parse_args()

    # Paths of the preprocessed data
    preprocessed_train_path = get_preprocessed_data_path("train")
    if(not os.path.isdir(preprocessed_train_path)):
       os.mkdir(preprocessed_train_path)
    preprocessed_test_path = get_preprocessed_data_path("test")
    if(not os.path.isdir(preprocessed_test_path)):
       os.mkdir(preprocessed_test_path)
    # mfcc files
    output_train = f"{preprocessed_train_path}/mfccs.csv"
    output_test = f"{preprocessed_test_path}/mfccs.csv"
    # Calculate mfccs
    calculate_mfcc(get_train_data_path(), output_train, n_samples=args.n_samples)
    calculate_mfcc(get_test_data_path(), output_test, n_samples=args.n_samples)
