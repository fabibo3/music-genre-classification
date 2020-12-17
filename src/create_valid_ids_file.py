# Calculate the valid filenames and store

import os
import librosa
import pickle
import numpy as np
from tqdm import tqdm


file_path = "/home/fabi/workspace_python/music-genre-classification/data/test/Test/"

files = os.listdir(file_path)
files = sorted(files)
valid_ids = []
not_valid = []
for f in tqdm(files, position=0, leave=True):
    fn_full = os.path.join(file_path, f)
    try:
        x, sr = librosa.load(fn_full)
        valid_ids.append(int(f.split(".")[0]))
    except:
        not_valid.append(f)

print('Files not valid: ', not_valid)
valid_ids = np.asarray(valid_ids)
out_file_name = "/home/fabi/workspace_python/music-genre-classification/data/test/preprocessed/valid_ids_sorted.pickle"
pickle.dump(valid_ids, open(out_file_name, "wb"))
