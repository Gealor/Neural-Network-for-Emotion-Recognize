"""Спикер-независимое разбиение корпуса на train/val/test.

Вся логика сплита в одном месте: сгруппировать файлы по спикерам, поделить
список спикеров, собрать файлы обратно.
"""
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

from domain_models import SplitConfig
from prepare_data.corpora import CorpusReader


def find_corpus_files(corpus: CorpusReader) -> list[Path]:
    """Файлы корпуса по его собственному glob (`CorpusReader.file_glob`).
    Модальность (.wav / .mp4 / ...) знает корпус, а не пайплайн."""
    return list(corpus.root.glob(corpus.file_glob))


def group_files_by_speaker(files_list: list[Path], corpus: CorpusReader) -> Tuple[dict, list]:
    """Группирует файлы по спикерам. Файлы, у которых не извлекается спикер или
    метка (напр. RAVDESS calm/surprised, TESS ps), отбрасываются."""
    files_by_speaker = defaultdict(list)
    for file_path in files_list:
        try:
            speaker = corpus.extract_speaker(file_path)
            corpus.extract_label(file_path)  # проверка распознаваемости файла
        except (ValueError, KeyError, IndexError) as e:
            print(f"Пропущен файл {file_path.name}: {e}")
            continue
        files_by_speaker[speaker].append(file_path)

    all_speaker_ids = sorted(files_by_speaker.keys())
    print(f"Найдено {len(all_speaker_ids)} спикеров.")
    return files_by_speaker, all_speaker_ids


def collect_files(speaker_list: list, files_by_speaker: dict) -> List[Path]:
    '''Собирает файлы всех спикеров из speaker_list в один список.'''
    file_list = []
    for speaker_id in speaker_list:
        file_list.extend(files_by_speaker[speaker_id])
    return file_list


def split_speakers(
    speakers: List[str],
    split_config: SplitConfig,
    rng: random.Random,
    shuffle: bool = True,
) -> Tuple[List[str], List[str], List[str]]:
    '''Делит список спикеров на train/val/test.'''
    ordered = speakers.copy()
    if shuffle:
        rng.shuffle(ordered)

    count = len(ordered)
    train_count = int(count * split_config.train)
    val_count = int(count * split_config.val)

    train_speakers = ordered[:train_count]
    val_speakers = ordered[train_count : train_count + val_count]
    test_speakers = ordered[train_count + val_count:]

    print(f"\nТренировочные спикеры ({len(train_speakers)}): {train_speakers}\n"
        f"Валидационные спикеры ({len(val_speakers)}): {val_speakers}\n"
        f"Тестовые спикеры ({len(test_speakers)}): {test_speakers}"
    )

    return train_speakers, val_speakers, test_speakers


def get_file_splits(
    corpus: CorpusReader,
    files_list: list[Path],
    split_config: SplitConfig,
    rng: random.Random,
) -> Tuple[List[Path], List[Path], List[Path]]:
    '''Группирует файлы по спикерам и разбивает на train/val/test.'''
    print("Подготовка данных и поиск файлов...")

    files_by_speaker, all_speaker_ids = group_files_by_speaker(files_list, corpus)

    if not all_speaker_ids:
        print("Категории не найдены. Возвращаем пустые наборы.")
        return [], [], []

    train_speakers, val_speakers, test_speakers = split_speakers(all_speaker_ids, split_config, rng)

    train_files = collect_files(train_speakers, files_by_speaker)
    val_files = collect_files(val_speakers, files_by_speaker)
    test_files = collect_files(test_speakers, files_by_speaker)

    print(f"\nНайдено файлов: Train - {len(train_files)}, Val - {len(val_files)}, Test - {len(test_files)}")
    return train_files, val_files, test_files
