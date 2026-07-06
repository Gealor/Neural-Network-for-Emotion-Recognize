from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np

from prepare_data.components.base import MediaPipeline
from prepare_data.components.data_splitter import ActorSplitter
from prepare_data.info_extractor import AbstractInfoExtractor

class DatasetProcessor:
    def __init__(self, info_extractor: AbstractInfoExtractor, pipeline: MediaPipeline, splitter: ActorSplitter):
        self.info_extractor = info_extractor
        self.pipeline = pipeline
        self.splitter = splitter

    def _find_files_by_actor(self, data_dir: Path) -> Tuple[dict, list]:
        """
        Ищет файлы поддерживаемых расширений (self.pipeline.file_extensions)
        и группирует их по дикторам, используя предоставленный экстрактор.
        """
        files_by_actor = defaultdict(list)
        extensions = set(self.pipeline.file_extensions)
        # Рекурсивный поиск всех .wav файлов в директории
        for file_path in data_dir.glob('**/*'):
            if file_path.suffix.lower() not in extensions:
                if file_path.is_file():
                    print(f"Пропущен файл {file_path.name}. Неподдерживаемый формат: {file_path.suffix}")
                continue
            try:
                # Экстрактор дает нам ID актера для группировки
                actor_id, _ = self.info_extractor.extract_info(file_path)
                files_by_actor[actor_id].append(file_path)
            except (ValueError, KeyError, IndexError) as e:
                # Игнорируем файлы, которые экстрактор не смог распознать
                print(f"Пропущен файл {file_path.name}: {e}")
                continue
        
        all_actor_ids = sorted(files_by_actor.keys())
        print(f"Найдено {len(all_actor_ids)} дикторов.")
        return files_by_actor, all_actor_ids

    
    def collect_files(self, actor_list, file_dict) -> List[Path]:
        '''
        Собирает все файлы всех дикторов из actor_list в один список. 
        Для каждого датасета (тренировочный, валидационный, тестовый) возвращает  новый список.
        '''
        file_list = []
        for actor_id in actor_list:
            file_list.extend(file_dict[actor_id])
        return file_list

    
    def _process_one_file(self, X: list, y: list, file: Path, augment: bool = False) -> None:
        '''Обработка одного файла датасета. ВНУТРЕННИЙ МЕТОД'''
        _, emotion_label = self.info_extractor.extract_info(file)
        features_gen = self.pipeline.process(file=file, augment=augment)

        for feature in features_gen:
            X.append(feature)
            y.append(emotion_label)


    def get_file_splits(self, data_dir: Path) -> Tuple[List[Path], List[Path], List[Path]]:
        '''
        Только собирает пути к файлам и разбивает их на train/val/test.
        '''
        print("Подготовка данных и поиск файлов...")
        files_by_actor, all_actors_ids = self._find_files_by_actor(data_dir=data_dir)

        if not all_actors_ids:
            print("Категории не найдены. Возвращаем пустые наборы.")
            return [], [], []

        train_actors, val_actors, test_actors = self.splitter.split(all_actors_ids)

        train_files = self.collect_files(train_actors, files_by_actor)
        val_files = self.collect_files(val_actors, files_by_actor)
        test_files = self.collect_files(test_actors, files_by_actor)

        print(f"\nНайдено файлов: Train - {len(train_files)}, Val - {len(val_files)}, Test - {len(test_files)}")
        return train_files, val_files, test_files


    def process_in_batches(self, files: List[Path], augment: bool = False, batch_size: int = 50):
        '''
        ГЕНЕРАТОР: Обрабатывает файлы пачками (батчами) и отдает их порциями, 
        чтобы не забивать оперативную память.
        '''
        X, y = [], []
        for i, file in enumerate(files, 1):
            self._process_one_file(X, y, file=file, augment=augment)
            
            # Как только обработали `batch_size` файлов - отдаем накопленное через yield
            if i % batch_size == 0:
                yield np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
                X, y = [], [] # Очищаем списки для следующей пачки
                
        # Отдаем "хвост" - оставшиеся файлы
        if X:
            yield np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
