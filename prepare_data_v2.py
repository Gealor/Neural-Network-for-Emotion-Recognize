import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from prepare_data.dataset_processor import DatasetProcessor
from prepare_data.info_extractor import CREMADExtractor, RAVDESSExtractor, TESSExtractor


ROOT_DATA_DIR = Path(__file__).parent / "dataset"
OUTPUT_DIR = Path(__file__).parent / "processed_data"

def temporary_process_and_safe(datasets_to_process: Dict[str, Dict[str, Any]]) -> None:
    for name, config in datasets_to_process.items():
        data_path = config["path"]
        extractor = config["extractor"]

        if not data_path.exists():
            print(f"Директория для датасета '{name}' не найдена по пути {data_path}. Пропускаем.")
            continue
        
        processor = DatasetProcessor(info_extractor=extractor)

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.processed_dataset(data_dir=data_path)

        temp_dir = OUTPUT_DIR / f"temp_{name}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        print(f"Сохраняем временные файлы для {name} в {temp_dir}...")

        np.save(temp_dir / "X_train.npy", np.array(X_train))
        np.save(temp_dir / "y_train.npy", np.array(y_train))
        np.save(temp_dir / "X_val.npy", np.array(X_val))
        np.save(temp_dir / "y_val.npy", np.array(y_val))
        np.save(temp_dir / "X_test.npy", np.array(X_test))
        np.save(temp_dir / "y_test.npy", np.array(y_test))

        del X_train, y_train, X_val, y_val, X_test, y_test

def concatenate_temporary_files(
    all_parts: Dict[str, Tuple[List, List]],
    datasets_to_process: Dict[str, Dict[str, Any]]
) -> None:
    for name in datasets_to_process.keys():
        temp_dir = OUTPUT_DIR / f"temp_{name}"
        if not temp_dir.exists(): 
            continue
        
        x_train_part = np.load(temp_dir / "X_train.npy")
        if x_train_part.shape[0] > 0:
            all_parts['train'][0].append(x_train_part)
            all_parts['train'][1].append(np.load(temp_dir / "y_train.npy"))
        
        x_val_part = np.load(temp_dir / "X_val.npy")
        if x_val_part.shape[0] > 0:
            all_parts['val'][0].append(x_val_part)
            all_parts['val'][1].append(np.load(temp_dir / "y_val.npy"))
            
        x_test_part = np.load(temp_dir / "X_test.npy")
        if x_test_part.shape[0] > 0:
            all_parts['test'][0].append(x_test_part)
            all_parts['test'][1].append(np.load(temp_dir / "y_test.npy"))


def main():
    datasets_to_process = {
        "RAVDESS": {
            "path": ROOT_DATA_DIR / "RAVDESS",
            "extractor": RAVDESSExtractor()
        },
        "TESS": {
            "path": ROOT_DATA_DIR / "TESS",
            "extractor": TESSExtractor()
        },
        "CREMA-D": {
            "path": ROOT_DATA_DIR / "CREMA-D",
            "extractor": CREMADExtractor()
        }
    }

    temporary_process_and_safe(datasets_to_process=datasets_to_process)
    
    all_parts = {'train': ([], []), 'val': ([], []), 'test': ([], [])}

    concatenate_temporary_files(all_parts=all_parts, datasets_to_process=datasets_to_process)
    
    X_train_final = np.concatenate(all_parts['train'][0], axis=0)
    y_train_final = np.concatenate(all_parts['train'][1], axis=0)

    if all_parts['val'][0]:
        X_val_final = np.concatenate(all_parts['val'][0], axis=0)
        y_val_final = np.concatenate(all_parts['val'][1], axis=0)
    else:
        X_val_final = np.empty((0, *X_train_final.shape[1:])) 
        y_val_final = np.empty((0,))

    if all_parts['test'][0]:
        X_test_final = np.concatenate(all_parts['test'][0], axis=0)
        y_test_final = np.concatenate(all_parts['test'][1], axis=0)
    else:
        X_test_final = np.empty((0, *X_train_final.shape[1:]))
        y_test_final = np.empty((0,))

    shuffle_indices = np.random.permutation(len(X_train_final))
    X_train_final = X_train_final[shuffle_indices]
    y_train_final = y_train_final[shuffle_indices]

    print("\nФинальные размеры объединенных данных:")
    print(f"Train: X={X_train_final.shape}, y={y_train_final.shape}")
    print(f"Val:   X={X_val_final.shape}, y={y_val_final.shape}")
    print(f"Test:  X={X_test_final.shape}, y={y_test_final.shape}")

    np.save(OUTPUT_DIR / "X_train.npy", X_train_final)
    np.save(OUTPUT_DIR / "y_train.npy", y_train_final)
    np.save(OUTPUT_DIR / "X_val.npy", X_val_final)
    np.save(OUTPUT_DIR / "y_val.npy", y_val_final)
    np.save(OUTPUT_DIR / "X_test.npy", X_test_final)
    np.save(OUTPUT_DIR / "y_test.npy", y_test_final)

    print(f"\nВсе данные успешно объединены и сохранены в директорию: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
    import calculate_stats
    
