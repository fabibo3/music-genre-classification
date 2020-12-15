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
import torch.utils.data
import pickle

class MusicDataset(torch.utils.data.Dataset):
    """
    Class containing the music data given as .mp3 files
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
                self._labels[file_no]-1


    def get_whole_dataset(self) -> (list, np.ndarray, dict):
        valid_labels = [self._labels[f] for f in self._valid_files]
        return self._valid_files, self._features, valid_labels

    def get_whole_dataset_labels_zero_based(self) -> (list, np.ndarray, dict):
        valid_labels = [self._labels[f]-1 for f in self._valid_files]
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


class MelSpectroDataset(torch.utils.data.Dataset):
    """
    Class representing a set of melspectrograms provided as .pickle file
    """
    def __init__(self, melspectro_file: str,
                 file_names_file: str=None,
                 label_file: str=None):
        """
        @param melspectro_file: The file containing melspectrograms for a set
        of music files. The file should be placed in the data directory root
        folder.
        @param file_names_file: A file containing the names of the files
        corresponding to the melspectros
        @param label_file: A file containing corresponding labels. 

        Attention: All files should contain an equal number of rows if not None
        and the order of represented files should be the same in each file!
        """
        self._root_dir_name = get_dataset_base_folder()
        if(file_names_file is not None):
            self._file_names_file = os.path.join(self._root_dir_name, file_names_file)
            self.contains_file_names = True
        else:
            self._file_names_file = None
            self.contains_file_names = False
        if(label_file is not None):
            self._label_file = os.path.join(self._root_dir_name, label_file)
            self.contains_labels = True
        else:
            self._label_file = None
            self.contains_labels = False
        self._data_file_name = os.path.join(self._root_dir_name,
                                            melspectro_file)

        # Load melspectros
        self.data = pickle.load(open(self._data_file_name, "rb"))
        self.data = np.expand_dims(self.data, axis=1) # torch dimensions
        self.datashape = self.data.shape
        # Load labels if possible
        if(self.contains_labels):
            self.labels = pickle.load( open(self._label_file, "rb" ))
            self.labels = np.argmax(self.labels, axis=1) # One-hot to integer
            assert(self.labels.shape[0] == self.data.shape[0])
        else:
            self.labels = None
        # Load filenames if possible
        if(self.contains_file_names):
            self.file_names = pickle.load(open(self._file_names_file, "rb"))
            assert(self.file_names.shape[0] == self.data.shape[0])
        else:
            self.file_names = None

    def __len__(self):
        return self.data.shape[0]

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

    def get_item_from_index(self, index: int) -> (int, np.ndarray, np.array):
        """
        Get a dataset element in the following form. If any of the information is not
        available, None is returned
        (filename, data, label)
        """
        if(self.contains_file_names):
            fn = self.file_names[index]
        else:
            fn = -1
        if(self.contains_labels):
            label = self.labels[index]
        else:
            label = -1
        return fn, self.data[index], label

    def set_subset(self, indices: list):
        """
        Set the dataset to a subset of samples
        """
        self.data = self.data[indices]
        if(self.contains_file_names):
            self.file_names = self.file_names[indices]
        if(self.contains_labels):
            label = self.labels[indices]

    def get_whole_dataset(self) -> (np.ndarray, np.ndarray,
                                    np.ndarray):
        return self.file_names, self.data, self.labels

