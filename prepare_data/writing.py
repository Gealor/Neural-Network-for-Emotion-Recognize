"""Потоковая запись подготовленных фич на диск.

Файлы обрабатываются пачками и дописываются в общие `X_{split}.npy` /
`y_{split}.npy` через `NpyAppendArray`, чтобы не держать весь датасет в памяти.
"""
import shutil
from pathlib import Path
from typing import List

import numpy as np
from npy_append_array import NpyAppendArray

import config
from domain_models import DatasetType
from prepare_data.corpora import CorpusReader
from prepare_data.pipelines.base import MediaPipeline


def recreate_folder(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def append_data_into_file(filepath: Path, data: np.ndarray) -> None:
    with NpyAppendArray(filepath) as npaa:
        npaa.append(data)


def save_dataset(X: np.ndarray, y: np.ndarray, type_dataset: DatasetType) -> None:
    X_path = config.OUTPUT_DIR / f"X_{type_dataset}.npy"
    y_path = config.OUTPUT_DIR / f"y_{type_dataset}.npy"
    if len(X) > 0:
        append_data_into_file(X_path, X)
        append_data_into_file(y_path, y)


def process_one_file(
    pipeline: MediaPipeline,
    emotion_label: int,
    X: list,
    y: list,
    file: Path,
    augment: bool = False,
) -> None:
    '''Обработка одного файла датасета. ВНУТРЕННИЙ МЕТОД'''
    features_gen = pipeline.process(file=file, augment=augment)

    for feature in features_gen:
        X.append(feature)
        y.append(emotion_label)


def process_in_batches(
    corpus: CorpusReader,
    pipeline: MediaPipeline,
    files: List[Path],
    augment: bool = False,
    batch_size: int = 50,
):
    '''
    ГЕНЕРАТОР: Обрабатывает файлы пачками (батчами) и отдает их порциями,
    чтобы не забивать оперативную память.
    '''
    X, y = [], []
    for i, file in enumerate(files, 1):
        emotion_label = corpus.extract_label(file)
        process_one_file(pipeline, emotion_label, X, y, file=file, augment=augment)

        # Как только обработали `batch_size` файлов - отдаем накопленное через yield
        if i % batch_size == 0:
            yield np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
            X, y = [], []  # Очищаем списки для следующей пачки

    # Отдаем "хвост" - оставшиеся файлы
    if X:
        yield np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def process_with_batching(
    corpus: CorpusReader,
    pipeline: MediaPipeline,
    files: List[Path],
    type_dataset: DatasetType,
    augment: bool = False,
    batch_size: int = config.BATCH_SIZE,
) -> None:
    print(f"Потоковая запись и сохранение {type_dataset} данных...")
    for X_batch, y_batch in process_in_batches(corpus, pipeline, files, augment=augment, batch_size=batch_size):
        save_dataset(X_batch, y_batch, type_dataset=type_dataset)


def check_and_create_empty_files() -> None:
    """Если для val или test не было данных, создаем пустые массивы"""
    train_x_file = config.OUTPUT_DIR / "X_train.npy"
    if not train_x_file.exists():
        return  # Если даже трейна нет, выходим

    # Загружаем заголовок X_train, чтобы узнать размерность фичей (не загружая в память)
    train_shape = np.load(train_x_file, mmap_mode='r').shape
    feature_shape = train_shape[1:]

    for ds_type in ('val', 'test'):
        x_file = config.OUTPUT_DIR / f"X_{ds_type}.npy"
        y_file = config.OUTPUT_DIR / f"y_{ds_type}.npy"

        if not x_file.exists():
            np.save(x_file, np.empty((0, *feature_shape), dtype=np.float32))
            np.save(y_file, np.empty((0,), dtype=np.float32))
