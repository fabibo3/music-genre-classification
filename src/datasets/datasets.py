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
import pandas as pd

class MusicDataset(object):
    """
    Class containing the music data
    """
    def __init__(self, split: str, mfcc_file: str="mfccs.csv", files=None,
                 features="adapte"):
        """
        @param split: "train" or "test"
        @param mfcc_file: Opionally, specify the name of the mfcc file
        @param files: Optional, predefine the files in the dataset
        @param features: "mfcc" or "adapte"
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
        if(files==None):
            self._all_files = sorted(os.listdir(self._file_dir))
        else:
            self._all_files = sorted(files)

        # Choose features
        if(features=="mfcc"):
            mfcc_file = os.path.join(get_preprocessed_data_path(self._split), mfcc_file)
            self._features, self._valid_files = self.get_mfccs(mfcc_file)
            self._feature_names = "mfcc only"
        elif(features=="adapte"):
            features_file = os.path.join(get_dataset_base_folder(),
                                                      "features_adapte.csv")
            feature_header_file = os.path.join(get_dataset_base_folder(),
                                               "features_head.csv")
            self._feature_names =\
                pd.read_csv(filepath_or_buffer=feature_header_file, sep=",")
            self._features, self._valid_files =\
                self.get_features_from_file(features_file)
        else:
            raise ValueError("Unknown features")


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
                self._features[index],\
                self._labels[file_no]

    def get_whole_dataset(self) -> (list, np.ndarray, dict):
        valid_labels = [self._labels[f] for f in self._valid_files]
        return self._valid_files, self._features, valid_labels

    def get_whole_dataset_as_pd(self) -> (list, pd.DataFrame, pd.DataFrame):
        valid_labels = [self._labels[f] for f in self._valid_files]
        return self._valid_files, self._pd_data, valid_labels


    def get_labels(self, label_file: str, split: str) -> dict:
        with open(label_file, 'r') as f:
            reader = csv.reader(f, delimiter=",")
            # Ignore first line
            next(reader)
            labels = {}
            for line in reader:
                if(split=="train"):
                    if(line[0]+".mp3" in self._all_files):
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
                if(line[0] in self._all_files):
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

    def get_features_from_file(self, features_file: str) -> (pd.DataFrame, list):
        """
        Load features from a file and return the data as panda DataFrame and
        the names of the valid files
        """
        label_data = pd.read_csv(filepath_or_buffer=self._label_file, sep=",")
        iter_csv = pd.read_csv(filepath_or_buffer=features_file, sep=",",
                               iterator=True, chunksize=10000)
        feature_data = pd.concat([chunk for chunk in iter_csv])

        ids = [f.split(".")[0] for f in self._all_files]
        label_data = label_data[label_data['track_id'].isin(ids)]
        data = pd.merge(label_data, feature_data,
                        on='track_id')

        if(self._split != "test"):
            data = data.drop('genre_id', axis=1)

        valid_files = [f"{v:06}" + ".mp3" for v in data['track_id'].values]

        self._pd_data = data.drop('track_id', axis=1)
        self._pd_labels = label_data.drop('track_id', axis=1)

        data = data.drop('track_id', axis=1)
        print(f"Shape of dataset: {data.shape}")

        return data.values, valid_files

