import gc
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np

import config
from domain_models import AudioConfig, PipelineConfig, PrepConfig, SplitConfig
from prepare_data.calculate_stats import calculate_norm_params
from prepare_data.files import get_all_files_by_format, get_file_splits
from prepare_data.info_extractor import (
    AbstractInfoExtractor,
    CREMADExtractor,
    RAVDESSExtractor,
    TESSExtractor,
)
from prepare_data.pipelines.audio.pipeline import build_audio_pipeline
from prepare_data.process_files_batching import process_with_batching

datasets_to_process = {
    "RAVDESS": {
        "path": config.ROOT_DATA_DIR / "RAVDESS",
        "extractor": RAVDESSExtractor(),
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

def split_dataset(
    dataset_name: str,
    data_path: Path,
    info_extractor: AbstractInfoExtractor,
    split: SplitConfig,
    pipeline_config: PipelineConfig,
    limit_per_corpus: int | None = None,
) -> Tuple[List[Path], List[Path], List[Path]]:
    print(f"\n--- Разбиение датасета {dataset_name} на множества ---")
    files_list = get_all_files_by_format(data_path, formats=pipeline_config.file_extensions)
    if limit_per_corpus is not None:
        # Только для Phase 0: детерминированный срез равномерным шагом,
        # чтобы smoke-прогон занимал секунды и при этом задевал всех спикеров корпуса
        ordered = sorted(files_list)
        stride = max(1, len(ordered) // limit_per_corpus)
        files_list = ordered[::stride][:limit_per_corpus]
        print(f"limit_per_corpus={limit_per_corpus}: оставлено {len(files_list)} файлов (шаг {stride})")
    train_files, val_files, test_files = get_file_splits(info_extractor, files_list, split, pipeline_config.rng)
    return train_files, val_files, test_files


def split_and_process_datasets(
    pipeline_config: PipelineConfig,
    split: SplitConfig,
    limit_per_corpus: int | None = None,
) -> None:
    for name, cfg in datasets_to_process.items():
        data_path = cfg["path"]
        extractor = cfg["extractor"]
        if not data_path.exists():
            print(f"Директория для датасета '{name}' не найдена по пути {data_path}. Пропускаем.")
            continue

        train_files, val_files, test_files = split_dataset(
            dataset_name=name,
            data_path=data_path,
            info_extractor=extractor,
            split=split,
            pipeline_config=pipeline_config,
            limit_per_corpus=limit_per_corpus,
        )

        pipeline = pipeline_config.pipeline
        process_with_batching(extractor, pipeline, train_files, "train", augment=True)
        process_with_batching(extractor, pipeline, val_files, "val")
        process_with_batching(extractor, pipeline, test_files, "test")

        gc.collect()


def check_and_create_empty_files():
    """Если для val или test не было данных, создаем пустые массивы"""
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


def build_default_config() -> PrepConfig:
    """Дефолтная конфигурация подготовки данных, собранная из config.py."""
    return PrepConfig(
        split=SplitConfig(train=config.TRAIN_SPLIT, val=config.VAL_SPLIT),
        audio=AudioConfig(
            n_mels=config.HEIGHT,
            max_pad_len=config.WIDTH,
            include_deltas=config.INCLUDE_DELTAS,
        ),
        limit_per_corpus=config.LIMIT_PER_CORPUS,
    )


def main(cfg: PrepConfig):
    recreate_folder(config.OUTPUT_DIR)

    # Один общий RNG на сплит и аугментацию (см. PrepConfig.seed)
    rng = random.Random(cfg.seed)
    pipeline = build_audio_pipeline(
        n_mels=cfg.audio.n_mels,
        max_pad_len=cfg.audio.max_pad_len,
        include_deltas=cfg.audio.include_deltas,
        rng=rng,
        augment_count=cfg.audio.augment_count,
    )
    pipeline_config = PipelineConfig(
        pipeline=pipeline,
        rng=rng
    )
    split_and_process_datasets(
        pipeline_config=pipeline_config,
        split=cfg.split,
        limit_per_corpus=cfg.limit_per_corpus,
    )

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

if __name__ == '__main__':
    main(build_default_config())
    calculate_norm_params(Path('processed_data/X_train.npy'))

