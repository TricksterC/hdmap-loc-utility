import pickle
from pathlib import Path

import yaml


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_pickle(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
