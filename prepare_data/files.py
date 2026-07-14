from collections import defaultdict
from pathlib import Path
import random
from typing import Iterable, List, Tuple

from domain_models import SplitConfig
from prepare_data.data_splitter import split_data
from prepare_data.info_extractor import AbstractInfoExtractor

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

def group_files_by_actor(files_list: list[Path], info_extractor: AbstractInfoExtractor) -> Tuple[dict, list]:
    """Группирует их по дикторам, используя предоставленный экстрактор."""
    files_by_actor = defaultdict(list)
    for file_path in files_list:
        try:
            # Экстрактор дает нам ID актера для группировки
            actor_id, _ = info_extractor.extract_info(file_path)
            files_by_actor[actor_id].append(file_path)
        except (ValueError, KeyError, IndexError) as e:
            # Игнорируем файлы, которые экстрактор не смог распознать
            print(f"Пропущен файл {file_path.name}: {e}")
            continue
    
    all_actor_ids = sorted(files_by_actor.keys())
    print(f"Найдено {len(all_actor_ids)} дикторов.")
    return files_by_actor, all_actor_ids


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
    info_extractor: AbstractInfoExtractor,
    files_list: list[Path], 
    split_config: SplitConfig,
    rng: random.Random
) -> Tuple[List[Path], List[Path], List[Path]]:
    '''
    Только собирает пути к файлам и разбивает их на train/val/test.
    '''
    print("Подготовка данных и поиск файлов...")

    files_by_actor, all_actors_ids = group_files_by_actor(files_list, info_extractor)

    if not all_actors_ids:
        print("Категории не найдены. Возвращаем пустые наборы.")
        return [], [], []

    train_actors, val_actors, test_actors = split_data(all_actors_ids, split_config, rng)

    train_files = collect_files(train_actors, files_by_actor)
    val_files = collect_files(val_actors, files_by_actor)
    test_files = collect_files(test_actors, files_by_actor)

    print(f"\nНайдено файлов: Train - {len(train_files)}, Val - {len(val_files)}, Test - {len(test_files)}")
    return train_files, val_files, test_files