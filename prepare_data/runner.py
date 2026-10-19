"""Оркестрация подготовки данных: конфиг -> сплит по корпусам -> запись -> статы."""
import gc
import random
from pathlib import Path
from typing import Generator, List, Tuple

import numpy as np

import config
from domain_models import AudioConfig, PrepConfig, SplitConfig
from prepare_data.corpora import CORPORA, CorpusReader
from prepare_data.pipelines.audio import AudioPipeline
from prepare_data.splitting import find_corpus_files, get_file_splits
from prepare_data.ssl_manifest import (
    ensure_empty_split_files,
    print_summary,
    save_split_files,
)
from prepare_data.stats import calculate_norm_params
from prepare_data.writing import (
    check_and_create_empty_files,
    process_with_batching,
    recreate_folder,
)


def build_default_config(seed: int = 42) -> PrepConfig:
    """Дефолтная конфигурация подготовки данных, собранная из config.py."""
    return PrepConfig(
        split=SplitConfig(train=config.TRAIN_SPLIT, val=config.VAL_SPLIT),
        audio=AudioConfig(
            n_mels=config.HEIGHT,
            max_pad_len=config.WIDTH,
            include_deltas=config.INCLUDE_DELTAS,
        ),
        seed=seed,
        limit_per_corpus=config.LIMIT_PER_CORPUS,
    )


def split_corpus(
    corpus_name: str,
    corpus: CorpusReader,
    split: SplitConfig,
    rng: random.Random,
    limit_per_corpus: int | None = None,
) -> Tuple[List[Path], List[Path], List[Path]]:
    print(f"\n--- Разбиение корпуса {corpus_name} на множества ---")
    files_list = find_corpus_files(corpus)
    if limit_per_corpus is not None:
        # Детерминированный срез равномерным шагом: быстрые smoke-прогоны и тесты
        # проходят по всем спикерам корпуса, но за секунды (см. config.LIMIT_PER_CORPUS)
        ordered = sorted(files_list)
        stride = max(1, len(ordered) // limit_per_corpus)
        files_list = ordered[::stride][:limit_per_corpus]
        print(f"limit_per_corpus={limit_per_corpus}: оставлено {len(files_list)} файлов (шаг {stride})")
    train_files, val_files, test_files = get_file_splits(corpus, files_list, split, rng)
    return train_files, val_files, test_files


def process_corpora(
    split: SplitConfig,
    rng: random.Random,
    limit_per_corpus: int | None = None,
) -> Generator[Tuple[CorpusReader, str, List[Path], List[Path], List[Path]]]:
    for name, corpus in CORPORA.items():
        if not corpus.root.exists():
            print(f"Директория корпуса '{name}' не найдена по пути {corpus.root}. Пропускаем.")
            continue

        train_files, val_files, test_files = split_corpus(
            corpus_name=name,
            corpus=corpus,
            split=split,
            rng=rng,
            limit_per_corpus=limit_per_corpus,
        )

        yield corpus, name, train_files, val_files, test_files


def _print_final_shapes() -> None:
    print("\nФинальные размеры объединенных данных:")
    for split_name in ("train", "val", "test"):
        try:
            # mmap_mode="r" позволяет мгновенно прочитать shape без загрузки массива в оперативную память
            X_shape = np.load(config.OUTPUT_DIR / f"X_{split_name}.npy", mmap_mode="r").shape
            y_shape = np.load(config.OUTPUT_DIR / f"y_{split_name}.npy", mmap_mode="r").shape
            print(f"{split_name.capitalize()}: X={X_shape}, y={y_shape}")
        except Exception:
            pass


def run(cfg: PrepConfig) -> None:
    recreate_folder(config.OUTPUT_DIR)

    # Один общий RNG на сплит и аугментацию (см. PrepConfig.seed)
    rng = random.Random(cfg.seed)
    pipeline = AudioPipeline(cfg.audio, rng)
    for corpus, name, train_files, val_files, test_files in process_corpora(
        split=cfg.split,
        rng=rng,
        limit_per_corpus=cfg.limit_per_corpus,
    ):
        process_with_batching(corpus, pipeline, train_files, "train", augment=True, group=name)
        process_with_batching(corpus, pipeline, val_files, "val", group=name)
        process_with_batching(corpus, pipeline, test_files, "test", group=name)

        gc.collect()

    check_and_create_empty_files()
    _print_final_shapes()

    print(f"\nВсе данные успешно объединены и сохранены в директорию: {config.OUTPUT_DIR}")
    calculate_norm_params(config.OUTPUT_DIR / "X_train.npy")


def run_ssl(cfg: PrepConfig) -> None:
    recreate_folder(config.SSL_OUTPUT_DIR)

    rng = random.Random(cfg.seed)
    for corpus, name, train_files, val_files, test_files in process_corpora(
        split=cfg.split,
        rng=rng,
        limit_per_corpus=cfg.limit_per_corpus,
    ):
        save_split_files(corpus, name, "train", train_files)
        save_split_files(corpus, name, "val", val_files)
        save_split_files(corpus, name, "test", test_files)

    ensure_empty_split_files()
    print_summary()
