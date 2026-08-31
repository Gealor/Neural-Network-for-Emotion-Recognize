from collections import defaultdict
from pathlib import Path
import random
from typing import Iterable, List, Tuple

from domain_models import SplitConfig
from prepare_data.corpora import CorpusReader
from prepare_data.data_splitter import split_data

def get_all_files_by_format(data_dir: Path, formats: Iterable[str]) -> list[Path]:
    """Ищет файлы поддерживаемых расширений"""
    print(f"Поддерживаемые расширения: {formats}")

    files = []
    for file_path in data_dir.glob('**/*'):
        if file_path.suffix.lower() not in formats:
            if file_path.is_file():
                print(f"Пропущен файл {file_path.name}. Неподдерживаемый формат: {file_path.suffix}")
            continue

        files.append(file_path)
        
    return files

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
    print(f"Найдено {len(all_speaker_ids)} дикторов.")
    return files_by_speaker, all_speaker_ids


def collect_files(actor_list, file_dict) -> List[Path]:
    '''
    Собирает все файлы всех дикторов из actor_list в один список. 
    Для каждого датасета (тренировочный, валидационный, тестовый) возвращает  новый список.
    '''
    file_list = []
    for actor_id in actor_list:
        file_list.extend(file_dict[actor_id])
    return file_list


def get_file_splits(
    corpus: CorpusReader,
    files_list: list[Path],
    split_config: SplitConfig,
    rng: random.Random
) -> Tuple[List[Path], List[Path], List[Path]]:
    '''
    Только собирает пути к файлам и разбивает их на train/val/test.
    '''
    print("Подготовка данных и поиск файлов...")

    files_by_speaker, all_speaker_ids = group_files_by_speaker(files_list, corpus)

    if not all_speaker_ids:
        print("Категории не найдены. Возвращаем пустые наборы.")
        return [], [], []

    train_actors, val_actors, test_actors = split_data(all_speaker_ids, split_config, rng)

    train_files = collect_files(train_actors, files_by_speaker)
    val_files = collect_files(val_actors, files_by_speaker)
    test_files = collect_files(test_actors, files_by_speaker)

    print(f"\nНайдено файлов: Train - {len(train_files)}, Val - {len(val_files)}, Test - {len(test_files)}")
    return train_files, val_files, test_files