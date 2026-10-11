"""Потоковая запись подготовленных фич на диск.

Файлы обрабатываются пачками и дописываются в общие `X_{split}.npy` /
`y_{split}.npy` через `NpyAppendArray`, чтобы не держать все данные в памяти.
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


def save_groups(count: int, type_dataset: DatasetType, group: str) -> None:
    """Метка корпуса на каждую строку сплита — для per-corpus метрик на тесте."""
    if count > 0:
        groups = np.full(count, group, dtype="<U16")
        append_data_into_file(config.OUTPUT_DIR / f"groups_{type_dataset}.npy", groups)


def save_paths(paths: np.ndarray, type_dataset: DatasetType) -> None:
    """Путь исходного wav-файла на каждую строку сплита (относительно
    config.ROOT_PATH — переносимо между машинами). Аугментированные копии
    одного файла делят один и тот же путь. Нужно для SSL-экспериментов
    (HuBERT/WavLM работают с сырым waveform, а не с X_*.npy спектрограммами)."""
    if len(paths) > 0:
        append_data_into_file(config.OUTPUT_DIR / f"paths_{type_dataset}.npy", paths)


def process_one_file(
    pipeline: MediaPipeline,
    emotion_label: int,
    X: list,
    y: list,
    paths: list,
    file: Path,
    augment: bool = False,
) -> None:
    '''Прогон одного файла через пайплайн: оригинал + аугментации в X, метка в y,
    путь файла в paths (один на каждую, в т.ч. аугментированную, копию).'''
    features_gen = pipeline.process(file=file, augment=augment)
    rel_path = str(file.relative_to(config.ROOT_PATH))

    for feature in features_gen:
        X.append(feature)
        y.append(emotion_label)
        paths.append(rel_path)


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
    X, y, paths = [], [], []
    for i, file in enumerate(files, 1):
        emotion_label = corpus.extract_label(file)
        process_one_file(pipeline, emotion_label, X, y, paths, file=file, augment=augment)

        # Как только обработали `batch_size` файлов - отдаем накопленное через yield
        if i % batch_size == 0:
            yield np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(paths, dtype="<U128")
            X.clear(); y.clear(); paths.clear()  # Очищаем списки для следующей пачки

    # Отдаем "хвост" - оставшиеся файлы
    if X:
        yield np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(paths, dtype="<U128")


def process_with_batching(
    corpus: CorpusReader,
    pipeline: MediaPipeline,
    files: List[Path],
    type_dataset: DatasetType,
    augment: bool = False,
    batch_size: int = config.BATCH_SIZE,
    group: str | None = None,
) -> None:
    print(f"Потоковая запись и сохранение {type_dataset} данных...")
    for X_batch, y_batch, paths_batch in process_in_batches(corpus, pipeline, files, augment=augment, batch_size=batch_size):
        save_dataset(X_batch, y_batch, type_dataset=type_dataset)
        save_paths(paths_batch, type_dataset=type_dataset)
        if group is not None:
            save_groups(len(X_batch), type_dataset=type_dataset, group=group)


def save_if_missing(path: Path, empty: np.ndarray) -> None:
    """Создаёт path с пустым массивом empty, если файла ещё нет."""
    if not path.exists():
        np.save(path, empty)


def check_and_create_empty_files() -> None:
    """Если для val или test не было данных, создаем пустые массивы"""
    train_x_file = config.OUTPUT_DIR / "X_train.npy"
    if not train_x_file.exists():
        return  # Если даже трейна нет, выходим

    # Загружаем заголовок X_train, чтобы узнать размерность фичей (не загружая в память)
    feature_shape = np.load(train_x_file, mmap_mode='r').shape[1:]

    for ds_type in ('val', 'test'):
        x_file = config.OUTPUT_DIR / f"X_{ds_type}.npy"
        # X и y всегда пишутся вместе (save_dataset), поэтому одной проверки по X достаточно
        if not x_file.exists():
            save_if_missing(x_file, np.empty((0, *feature_shape), dtype=np.float32))
            save_if_missing(config.OUTPUT_DIR / f"y_{ds_type}.npy", np.empty((0,), dtype=np.float32))
        save_if_missing(config.OUTPUT_DIR / f"groups_{ds_type}.npy", np.empty((0,), dtype="<U16"))
        save_if_missing(config.OUTPUT_DIR / f"paths_{ds_type}.npy", np.empty((0,), dtype="<U128"))
