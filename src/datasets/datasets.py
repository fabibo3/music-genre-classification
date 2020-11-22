"""
Wrapper for dataset
"""
__author__ = "Fabian Bongratz"

from utils.folders import get_dataset_base_folder,\
                            get_train_data_path,\
                            get_test_data_path,\
                            get_preprocessed_data_path 
import csv
import os
import numpy as np

class MusicDataset(object):
    """
    Class containing the music data
    """
    def __init__(self, split: str, mfcc_file: str="mfccs.csv"):
        """
        @param split: "train" or "test"
        """
        self._root_dir_name = get_dataset_base_folder()
        self._genres_file = self._root_dir_name + "/genres.csv"
        self._genres = self.get_genres(self._genres_file)
        self._split = split
        if(self._split=="train"):
            self._file_dir = get_train_data_path()
            self._label_file = os.path.join(self._root_dir_name, "train.csv")
        elif(self._split=="test"):
            self._file_dir = get_test_data_path()
            self._label_file = os.path.join(self._root_dir_name, "test.csv")
        else:
            raise ValueError("Split not recognized!")
        self._all_files = sorted(os.listdir(self._file_dir))
        self._mfcc_file = os.path.join(get_preprocessed_data_path(self._split), mfcc_file)
        self._mfccs, self._valid_files = self.get_mfccs(self._mfcc_file)
        self._labels = self.get_labels(self._label_file, self._split)

    def __len__(self):
        return len(self._valid_files)

    def __getitem__(self, key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            # get the data from direct index
            return self.get_item_from_index(key)
        else:
            raise TypeError("Invalid argument type.")

    def get_item_from_index(self, index: int) -> (str, np.ndarray, int):
        """
        Get a dataset element in the form
        (filename, mfcc coefficients, label)
        If self._split is equal to "test", the label will be -1
        """
        file_no = self._valid_files[index]
        return file_no,\
                self._mfccs[index],\
                self._labels[file_no]

    def get_whole_dataset(self) -> (list, np.ndarray, dict):
        valid_labels = [self._labels[f] for f in self._valid_files]
        return self._valid_files, self._mfccs, valid_labels

    def get_labels(self, label_file: str, split: str) -> dict:
        with open(label_file, 'r') as f:
            reader = csv.reader(f, delimiter=",")
            # Ignore first line
            next(reader)
            labels = {}
            for line in reader:
                if(split=="train"):
                    labels[line[0]+".mp3"] = int(line[1])
                else:
                    labels[line[0]+".mp3"] = -1

        return labels


    def get_mfccs(self, mfcc_file: str) -> (np.ndarray, list):
        """
        Read the preprocessed mfcc file and get a matrix of shape
        num_samples x mfcc_size as well as a list of all valid files
        """
        with open(mfcc_file, "r") as f:
            reader = csv.reader(f, delimiter=",")
            n_mfcc = next(reader)[-1].split("-")[-1]
            mfccs = []
            valid_files = []
            for line in reader:
                valid_files.append(line[0])
                mfccs.append([float(x) for x in line[1:]])

        sorted_indices = np.argsort(valid_files)
        valid_files = [valid_files[s] for s in sorted_indices]
        mfccs = np.asarray(mfccs)
        mfccs = mfccs[sorted_indices]

        return mfccs, valid_files

    def get_all_files(self) -> list:
        return self._all_files

    def get_genres(self, genres_file) -> int:
        """
        Get the number of different classes
        """
        genres = {}
        with open(genres_file, 'r') as f:
            reader = csv.reader(f, delimiter=",")
            # Ignore first row
            next(reader)
            for row in reader:
                genres[int(row[0])] = row[1]

        return genres

