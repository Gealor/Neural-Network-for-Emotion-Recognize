from pathlib import Path

import numpy as np
from npy_append_array import NpyAppendArray

import config
from domain_models import DatasetType


def append_data_into_file(filepath: Path, data: np.ndarray) -> None:
    with NpyAppendArray(filepath) as npaa:
        npaa.append(data)


def save_dataset(X: np.ndarray, y: np.ndarray, type_dataset: DatasetType) -> None:
    X_path = config.OUTPUT_DIR / f"X_{type_dataset}.npy"
    y_path = config.OUTPUT_DIR / f"y_{type_dataset}.npy"
    if len(X) > 0:
        append_data_into_file(X_path, X)
        append_data_into_file(y_path, y)
