from pathlib import Path
from typing import List

import numpy as np

import config
from prepare_data.corpora import CorpusReader
from prepare_data.writing import append_data_into_file, save_if_missing

SPLITS = ("train", "val", "test")


def save_split_files(
    corpus: CorpusReader,
    group: str,
    type_dataset: str,
    files: List[Path],
) -> None:
    """Путь (относительно config.ROOT_PATH), метка и корпус — одна строка на файл."""
    if not files:
        return

    paths = np.array([str(f.relative_to(config.ROOT_PATH)) for f in files], dtype="<U128")
    labels = np.array([corpus.extract_label(f) for f in files], dtype=np.float32)
    groups = np.full(len(files), group, dtype="<U16")

    append_data_into_file(config.SSL_OUTPUT_DIR / f"paths_{type_dataset}.npy", paths)
    append_data_into_file(config.SSL_OUTPUT_DIR / f"y_{type_dataset}.npy", labels)
    append_data_into_file(config.SSL_OUTPUT_DIR / f"groups_{type_dataset}.npy", groups)


def ensure_empty_split_files() -> None:
    """Если ни один корпус не дал файлов в какой-то сплит — создаём пустые
    массивы (аналог writing.check_and_create_empty_files для манифеста)."""
    for ds_type in SPLITS:
        save_if_missing(config.SSL_OUTPUT_DIR / f"paths_{ds_type}.npy", np.empty((0,), dtype="<U128"))
        save_if_missing(config.SSL_OUTPUT_DIR / f"y_{ds_type}.npy", np.empty((0,), dtype=np.float32))
        save_if_missing(config.SSL_OUTPUT_DIR / f"groups_{ds_type}.npy", np.empty((0,), dtype="<U16"))


def print_summary() -> None:
    print("\nМанифест для SSL-энкодеров (без аугментации):")
    for ds_type in SPLITS:
        path_file = config.SSL_OUTPUT_DIR / f"paths_{ds_type}.npy"
        if not path_file.exists():
            continue
        paths = np.load(path_file, mmap_mode='r')
        print(f"  {ds_type.capitalize()}: {len(paths)} записей")
