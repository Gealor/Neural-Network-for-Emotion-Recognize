from collections import defaultdict
from pathlib import Path
import random
from typing import Tuple

import librosa
import numpy as np

from prepare_data import utils
from prepare_data.info_extractor import AbstractInfoExtractor

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15

OUTPUT_DIR = 'processed_data'

class DatasetProcessor:
    def __init__(self, info_extractor: AbstractInfoExtractor):
        self.info_extractor = info_extractor


    def _find_files_by_actor(self, data_dir: Path) -> Tuple[dict, list]:
        """
        Универсальный метод: ищет все .wav файлы и группирует их по дикторам,
        используя предоставленный экстрактор.
        """
        files_by_actor = defaultdict(list)
        # Рекурсивный поиск всех .wav файлов в директории
        for file_path in data_dir.glob('**/*.wav'):
            try:
                # Экстрактор дает нам ID актера для группировки
                actor_id, _ = self.info_extractor.extract_info(file_path)
                files_by_actor[actor_id].append(file_path)
            except (ValueError, KeyError, IndexError) as e:
                # Игнорируем файлы, которые экстрактор не смог распознать
                print(f"Пропущен файл {file_path.name}: {e}")
                continue
        
        all_actor_ids = sorted(files_by_actor.keys())
        random.shuffle(all_actor_ids)
        print(f"Найдено {len(all_actor_ids)} дикторов.")
        return files_by_actor, all_actor_ids

    def _split_actors(self, all_actors_ids):
        num_actors = len(all_actors_ids)
        train_actors_count = int(num_actors * TRAIN_SPLIT)
        val_actors_count = int(num_actors * VAL_SPLIT)

        train_actors = all_actors_ids[:train_actors_count]
        val_actors = all_actors_ids[train_actors_count : train_actors_count + val_actors_count]
        test_actors = all_actors_ids[train_actors_count + val_actors_count:]

        print(f"\nТренировочные дикторы ({len(train_actors)}): {train_actors}")
        print(f"Валидационные дикторы ({len(val_actors)}): {val_actors}")
        print(f"Тестовые дикторы ({len(test_actors)}): {test_actors}")

        return train_actors, val_actors, test_actors
    
    def collect_files(self, actor_list, file_dict):
        file_list = []
        for actor_id in actor_list:
            file_list.extend(file_dict[actor_id])
        return file_list
    
    def _process_files(self, files: list, augment: bool = False):
        X, y = [], []
        for file in files:
            _, emotion_label = self.info_extractor.extract_info(file)
            audio, sr = librosa.load(str(file), sr=16000)

            # Оригинал + SpecAugment
            features_original = utils.extract_features(audio, sr)
            X.append(features_original)
            y.append(emotion_label)
        

            if augment:
                # Аугментация... (1x)
                for _ in range(2): # Добавляем 2 аугментированных копии
                    choice = random.choice(['noise', 'pitch', 'stretch', 'shift'])
                    if choice == 'noise':
                        aug_audio = utils.add_noise(audio)
                    elif choice == 'pitch':
                        aug_audio = utils.pitch_shift(audio, sr, n_steps=random.uniform(-2, 2))
                    elif choice == 'stretch':
                        rate = random.uniform(0.8, 1.2)
                        aug_audio = utils.time_stretch(audio, rate=rate)
                    else: # shift
                        aug_audio = utils.time_shift(audio)
                    
                    features_aug = utils.extract_features(aug_audio, sr)
                    features_aug_spec = utils.time_masking(utils.frequency_masking(features_aug))

                    X.append(features_aug_spec)
                    y.append(emotion_label)
                    
        return X, y
    

    def processed_dataset(self, data_dir: Path):
        print("Подготовка данных...")
        files_by_actor, all_actors_ids = self._find_files_by_actor(data_dir=data_dir)

        if not all_actors_ids:
            print("Дикторы не найдены. Возвращаем пустые наборы.")
            empty_result = (np.array([]), np.array([]))
            return empty_result, empty_result, empty_result

        train_actors, val_actors, test_actors = self._split_actors(all_actors_ids)

        train_files = self.collect_files(train_actors, files_by_actor)
        val_files = self.collect_files(val_actors, files_by_actor)
        test_files = self.collect_files(test_actors, files_by_actor)

        print(f"\n--- Обработка тренировочного набора ({len(train_files)} файлов) ---")
        X_train, y_train = self._process_files(train_files, augment=True)

        print(f"\n--- Обработка валидационного набора ({len(val_files)} файлов) ---")
        X_val, y_val = self._process_files(val_files)

        print(f"\n--- Обработка тестового набора ({len(test_files)} файлов) ---")
        X_test, y_test = self._process_files(test_files)

        print(f"\nСобрано {len(train_files) + len(val_files) + len(test_files)} исходных файлов.")
        print(f"Всего признаков после аугментации: {len(X_train) + len(X_val) + len(X_test)}")
        print(f"Тренировочный: {len(X_train)}, Валидационный: {len(X_val)}, Тестовый: {len(X_test)}")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# X_train, y_train = np.array(X_train), np.array(y_train)
# X_val, y_val = np.array(X_val), np.array(y_val)
# X_test, y_test = np.array(X_test), np.array(y_test)

# shuffle_indices = np.random.permutation(len(X_train))
# X_train = X_train[shuffle_indices]
# y_train = y_train[shuffle_indices]
