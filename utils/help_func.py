import pickle
import os


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        pkl_obj = pickle.load(f)
    return pkl_obj


def save_pickle(file_path, obj, protocol=3):
    parent_path = os.path.split(file_path)[0]
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f, protocol=protocol)
