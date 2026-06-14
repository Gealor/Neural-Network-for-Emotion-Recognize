import gc
from pathlib import Path
import shutil
from typing import List, Literal, Tuple
from warnings import deprecated

from npy_append_array import NpyAppendArray
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

def append_data_into_file(filepath: Path, data: np.ndarray) -> None:
    with NpyAppendArray(filepath) as npaa:
        npaa.append(data)

def save_dataset(X: np.ndarray, y: np.ndarray, type_dataset: Literal["train", "test", "val"]) -> None:
    X_path = config.OUTPUT_DIR / f"X_{type_dataset}.npy"
    y_path = config.OUTPUT_DIR / f"y_{type_dataset}.npy"
    if len(X) > 0:
        append_data_into_file(X_path, X)
        append_data_into_file(y_path, y)


def process_with_batching(
    processor: DatasetProcessor,
    files: List[Path],
    type_dataset: Literal["train", "test", "val"],
    augment: bool = False,
    batch_size: int = config.BATCH_SIZE,
):
    if files:
        print(f"Потоковая запись и сохранение {type_dataset} данных...")
        for X_batch, y_batch in processor.process_in_batches(files, augment=augment, batch_size=batch_size):
            save_dataset(X_batch, y_batch, type_dataset=type_dataset)


def process_and_accumulate() -> None:
    for name, cfg in datasets_to_process.items():
        data_path = cfg["path"]
        extractor = cfg["extractor"]

        if not data_path.exists():
            print(f"Директория для датасета '{name}' не найдена по пути {data_path}. Пропускаем.")
            continue
        
        processor = DatasetProcessor(info_extractor=extractor)

        print(f"\n--- Разбиение датасета {name} на множества ---")
        train_files, val_files, test_files = processor.get_file_splits(data_path)

        process_with_batching(processor, train_files, "train", augment=True)
        process_with_batching(processor, val_files, "val")
        process_with_batching(processor, test_files, "test")

        gc.collect()

@deprecated("This method is deprecated, use process_and_accumulate() instead this")
def process_and_accumulate_old() -> None:
    """Обрабатывает датасеты и сразу дописывает (append) их в итоговые .npy файлы."""
    
    for name, cfg in datasets_to_process.items():
        data_path = cfg["path"]
        extractor = cfg["extractor"]

        if not data_path.exists():
            print(f"Директория для датасета '{name}' не найдена по пути {data_path}. Пропускаем.")
            continue
        
        processor = DatasetProcessor(info_extractor=extractor)

        print(f"\nОбработка датасета {name}...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.processed_dataset(data_dir=data_path)

        print(f"Добавляем данные {name} напрямую в итоговые .npy файлы...")

        save_dataset(X_train, y_train, type_dataset="train")
        save_dataset(X_val, y_val, type_dataset="val")
        save_dataset(X_test, y_test, type_dataset="test")

        del X_train, y_train, X_val, y_val, X_test, y_test
        gc.collect()


def check_and_create_empty_files():
    """Если для val или test не было данных, создаем пустые массивы (как в вашем старом коде)."""
    train_x_file = config.OUTPUT_DIR / "X_train.npy"
    if not train_x_file.exists():
        return # Если даже трейна нет, выходим

    # Загружаем заголовок X_train, чтобы узнать размерность фичей (не загружая в память)
    train_shape = np.load(train_x_file, mmap_mode='r').shape
    feature_shape = train_shape[1:]

    for ds_type in ('val', 'test'):
        x_file = config.OUTPUT_DIR / f"X_{ds_type}.npy"
        y_file = config.OUTPUT_DIR / f"y_{ds_type}.npy"
        
        if not x_file.exists():
            np.save(x_file, np.empty((0, *feature_shape), dtype=np.float32))
            np.save(y_file, np.empty((0,), dtype=np.float32))


def recreate_folder(path: Path):
    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)

def main():
    recreate_folder(config.OUTPUT_DIR)
    
    process_and_accumulate()

    check_and_create_empty_files()

    print("\nФинальные размеры объединенных данных:")
    for dataset_type in ("train", "val", "test"):
        try:
            # mmap_mode="r" позволяет мгновенно прочитать shape без загрузки массива в оперативную память
            X_shape = np.load(config.OUTPUT_DIR / f"X_{dataset_type}.npy", mmap_mode="r").shape
            y_shape = np.load(config.OUTPUT_DIR / f"y_{dataset_type}.npy", mmap_mode="r").shape
            print(f"{dataset_type.capitalize()}: X={X_shape}, y={y_shape}")
        except Exception:
            pass

    print(f"\nВсе данные успешно объединены и сохранены в директорию: {config.OUTPUT_DIR}")

def calculate_norm_params():
    print("Расчет mean и std по каналам...")
    X = np.load('processed_data/X_train.npy', mmap_mode='r')
    
    # Определяем количество каналов
    # X.shape обычно (N, H, W) или (N, H, W, C)
    if len(X.shape) == 3:
        num_channels = 1
    else:
        num_channels = X.shape[-1]

    # Инициализируем накопители как float64 для точности
    sums = np.zeros(num_channels, dtype=np.float64)
    sums_sq = np.zeros(num_channels, dtype=np.float64)
    count = 0
    
    batch_size = 500 # Считаем кусками по 500 файлов для скорости
    num_samples = len(X)

    for i in range(0, num_samples, batch_size):
        end = min(i + batch_size, num_samples)
        batch = X[i:end].astype(np.float64) # Берем кусок данных
        
        # Если каналов несколько, считаем по осям (N, H, W)
        if num_channels > 1:
            sums += np.sum(batch, axis=(0, 1, 2))
            sums_sq += np.sum(batch**2, axis=(0, 1, 2))
            # Количество элементов в одном канале куска
            count += (end - i) * X.shape[1] * X.shape[2]
        else:
            sums[0] += np.sum(batch)
            sums_sq[0] += np.sum(batch**2)
            count += batch.size

    # Финальные расчеты
    mean = sums / count
    # std = sqrt( E[X^2] - (E[X])^2 )
    std = np.sqrt((sums_sq / count) - (mean**2))

    # Сохраняем (приводим к float32 для компактности)
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    # Меняем форму для удобного вычитания в генераторе: (1, 1, 1, C)
    if num_channels > 1:
        mean = mean.reshape(1, 1, 1, num_channels)
        std = std.reshape(1, 1, 1, num_channels)
    else:
        # Для 1 канала или если данные 3D
        mean = mean.reshape(1, 1, 1)
        std = std.reshape(1, 1, 1)

    np.save('processed_data/mean.npy', mean)
    np.save('processed_data/std.npy', std)

    print("Расчет окончен.")
    print("Mean per channel:", mean.flatten())
    print("Std per channel:", std.flatten())

if __name__ == '__main__':
    main()
    calculate_norm_params()

