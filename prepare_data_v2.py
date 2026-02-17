import gc
from pathlib import Path
import shutil
from typing import Dict, List, Literal, Tuple

import numpy as np

import config
from prepare_data.dataset_processor import DatasetProcessor
from prepare_data.info_extractor import CREMADExtractor, RAVDESSExtractor, TESSExtractor


datasets_to_process = {
    "RAVDESS": {
        "path": config.ROOT_DATA_DIR / "RAVDESS",
        "extractor": RAVDESSExtractor()
    },
    "TESS": {
        "path": config.ROOT_DATA_DIR / "TESS",
        "extractor": TESSExtractor()
    },
    "CREMA-D": {
        "path": config.ROOT_DATA_DIR / "CREMA-D",
        "extractor": CREMADExtractor()
    }
}

def _fill_datas_for_type(
    all_parts: Dict[str, Tuple[List, List]],
    temp_dir: Path,
    type_dataset: Literal["train", "test", "val"],
):
    '''Заполнение данных для конкретного набора данных (train, test, val).'''

    x_file = temp_dir / f"X_{type_dataset}.npy"
    y_file = temp_dir / f"y_{type_dataset}.npy"

    x_part = np.load(x_file, mmap_mode="r")
    if x_part.shape[0] > 0:
        all_parts[type_dataset][0].append(x_part)
        y_part = np.load(y_file, mmap_mode="r")
        all_parts[type_dataset][1].append(y_part)

# TODO: подумать, как можно оптимизировать сохранение массивов и уменьшить нагрузку на RAM (присмотреться к memmap)
def temporary_process_and_save() -> None:
    for name, cfg in datasets_to_process.items():
        data_path = cfg["path"]
        extractor = cfg["extractor"]

        if not data_path.exists():
            print(f"Директория для датасета '{name}' не найдена по пути {data_path}. Пропускаем.")
            continue
        
        processor = DatasetProcessor(info_extractor=extractor)

        print(f"\nОбработка датасета {name}...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.processed_dataset(data_dir=data_path)

        temp_dir = config.OUTPUT_DIR / f"temp_{name}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        print(f"Сохраняем временные файлы для {name} в {temp_dir}...")

        np.save(temp_dir / "X_train.npy", np.array(X_train, dtype=np.float32))
        np.save(temp_dir / "y_train.npy", np.array(y_train, dtype=np.float32))
        np.save(temp_dir / "X_val.npy", np.array(X_val, dtype=np.float32))
        np.save(temp_dir / "y_val.npy", np.array(y_val, dtype=np.float32))
        np.save(temp_dir / "X_test.npy", np.array(X_test, dtype=np.float32))
        np.save(temp_dir / "y_test.npy", np.array(y_test, dtype=np.float32))

        del X_train, y_train, X_val, y_val, X_test, y_test
        gc.collect()


def concatenate_temporary_files(
    all_parts: Dict[str, Tuple[List, List]],
) -> None:
    for name in datasets_to_process.keys():
        print(f"Объединяем {name}...")
        temp_dir = config.OUTPUT_DIR / f"temp_{name}"
        if not temp_dir.exists(): 
            print(f"Директория {temp_dir} не найдена. Пропускаем...")
            continue
        
        _fill_datas_for_type(all_parts, temp_dir, "train")
        _fill_datas_for_type(all_parts, temp_dir, "val")
        _fill_datas_for_type(all_parts, temp_dir, "test")

def cleanup_temp_files():
    """Удаляет временные папки после объединения."""
    print("Удаление временных файлов...")
    for name in datasets_to_process.keys():
        temp_dir = config.OUTPUT_DIR / f"temp_{name}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def clear_folder(path: Path):
    if path.exists():
        shutil.rmtree(path)

def main():
    clear_folder(config.OUTPUT_DIR)
    try:
        temporary_process_and_save()
        
        all_parts = {'train': ([], []), 'val': ([], []), 'test': ([], [])}

        concatenate_temporary_files(all_parts=all_parts)

        # схлопываем, т.к. в all_parts[<mark>][0/1] лежит список из списков данных для каждого из датасетов datasets_to_process, 
        # т.е. [[данные для RAVDESS], [данные для TESS], [данные для CREMA-D]]
        X_train_final = np.concatenate(all_parts['train'][0], axis=0) # схлопываем по первой оси, т.е. количеству файлов
        y_train_final = np.concatenate(all_parts['train'][1], axis=0) # схлопываем по первой оси, т.е. количеству файлов

        if all_parts['val'][0]:
            X_val_final = np.concatenate(all_parts['val'][0], axis=0)
            y_val_final = np.concatenate(all_parts['val'][1], axis=0)
        else:
            # если данных нет, для типа val, то создаем пустые массивы
            X_val_final = np.empty((0, *X_train_final.shape[1:])) 
            y_val_final = np.empty((0,))

        if all_parts['test'][0]:
            X_test_final = np.concatenate(all_parts['test'][0], axis=0)
            y_test_final = np.concatenate(all_parts['test'][1], axis=0)
        else:
            # # если данных нет, для типа test, то создаем пустые массивы
            X_test_final = np.empty((0, *X_train_final.shape[1:]))
            y_test_final = np.empty((0,))

        # Удаляю all_parts, т.к. он больше не нужен
        del all_parts 
        gc.collect()
    finally:
        cleanup_temp_files()


    print("\nФинальные размеры объединенных данных:")
    print(f"Train: X={X_train_final.shape}, y={y_train_final.shape}")
    print(f"Val:   X={X_val_final.shape}, y={y_val_final.shape}")
    print(f"Test:  X={X_test_final.shape}, y={y_test_final.shape}")

    np.save(config.OUTPUT_DIR / "X_train.npy", X_train_final)
    np.save(config.OUTPUT_DIR / "y_train.npy", y_train_final)
    np.save(config.OUTPUT_DIR / "X_val.npy", X_val_final)
    np.save(config.OUTPUT_DIR / "y_val.npy", y_val_final)
    np.save(config.OUTPUT_DIR / "X_test.npy", X_test_final)
    np.save(config.OUTPUT_DIR / "y_test.npy", y_test_final)

    print(f"\nВсе данные успешно объединены и сохранены в директорию: {config.OUTPUT_DIR}")

if __name__ == '__main__':
    main()
