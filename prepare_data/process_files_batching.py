from pathlib import Path
from typing import List

import numpy as np

import config
from domain_models import DatasetType
from prepare_data.pipelines.base import MediaPipeline
from prepare_data.info_extractor import AbstractInfoExtractor
from prepare_data.save_dataset import save_dataset


def process_one_file(
    pipeline: MediaPipeline, 
    emotion_label: int, 
    X: list, 
    y: list, 
    file: Path, 
    augment: bool = False
) -> None:
    '''Обработка одного файла датасета. ВНУТРЕННИЙ МЕТОД'''
    features_gen = pipeline.process(file=file, augment=augment)

    for feature in features_gen:
        X.append(feature)
        y.append(emotion_label)


def process_in_batches(
    info_extractor: AbstractInfoExtractor, 
    pipeline: MediaPipeline,
    files: List[Path], 
    augment: bool = False, 
    batch_size: int = 50
):
    '''
    ГЕНЕРАТОР: Обрабатывает файлы пачками (батчами) и отдает их порциями, 
    чтобы не забивать оперативную память.
    '''
    X, y = [], []
    for i, file in enumerate(files, 1):
        _, emotion_label = info_extractor.extract_info(file)
        process_one_file(pipeline, emotion_label, X, y, file=file, augment=augment)
        
        # Как только обработали `batch_size` файлов - отдаем накопленное через yield
        if i % batch_size == 0:
            yield np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
            X, y = [], [] # Очищаем списки для следующей пачки
            
    # Отдаем "хвост" - оставшиеся файлы
    if X:
        yield np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def process_with_batching(
    info_extractor: AbstractInfoExtractor,
    pipeline: MediaPipeline,
    files: List[Path],
    type_dataset: DatasetType,
    augment: bool = False,
    batch_size: int = config.BATCH_SIZE,
):
    print(f"Потоковая запись и сохранение {type_dataset} данных...")
    for X_batch, y_batch in process_in_batches(info_extractor, pipeline, files, augment=augment, batch_size=batch_size):
        save_dataset(X_batch, y_batch, type_dataset=type_dataset)