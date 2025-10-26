from collections import defaultdict
import os
import random
import librosa
import numpy as np
import pathlib

from sklearn.model_selection import train_test_split

DATA_DIR = pathlib.Path(__file__).parent / 'dataset'
OUTPUT_DIR = 'processed_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}


EMOTIONS_TO_NUM = {
    'neutral': 0,
    'calm': 1,
    'happy': 2,
    'sad': 3,
    'angry': 4,
    'fearful': 5,
    'disgust': 6,
    'surprised': 7
}

def add_noise(audio, noise_factor=0.005):
    """Добавляет случайный Гауссовский шум."""
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio.astype(type(audio[0]))

def pitch_shift(audio, sr, n_steps=2.0):
    """Сдвигает высоту тона аудиосигнала."""
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

def time_stretch(audio, rate=1.0):
    """
    Растягивает или сжимает аудио по времени без изменения высоты тона.
    - rate > 1.0: ускоряет аудио (делает его короче)
    - rate < 1.0: замедляет аудио (делает его длиннее)
    """
    return librosa.effects.time_stretch(y=audio, rate=rate)

def time_shift(audio, shift_max_ratio=0.2):
    """
    Сдвигает аудио во времени (циклически).
    Часть аудио, "выходящая" за один конец, появляется с другого.
    """
    shift_max = int(len(audio) * shift_max_ratio)
    shift_amount = np.random.randint(-shift_max, shift_max)
    return np.roll(audio, shift_amount)


def exctract_features(audio, sr, n_mels=128, max_pad_len=200):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=n_mels)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    # нормализация длины (обрезаю/дополняю нулями)
    if log_mel_spectrogram.shape[1] < max_pad_len:
        pad_width = max_pad_len - log_mel_spectrogram.shape[1]
        log_mel_spectrogram = np.pad(log_mel_spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        log_mel_spectrogram = log_mel_spectrogram[:, :max_pad_len]
    return log_mel_spectrogram

# SpecAugment, изменяет частотные полосы, чтобы менять тембр голоса и избежать привыкание модели к тембру
def frequency_masking(spectrogram, F=15, num_masks=1): # F - максимальная ширина маски
    cloned = spectrogram.copy()
    num_mel_channels = cloned.shape[0]
    for _ in range(num_masks):
        f = int(np.random.uniform(0.0, F))
        f0 = random.randint(0, num_mel_channels - f)
        cloned[f0:f0+f, :] = 0
    return cloned

def time_masking(spectrogram, T=25, num_masks=1): # T - максимальная ширина маски
    cloned = spectrogram.copy()
    len_spectro = cloned.shape[1]
    for _ in range(num_masks):
        t = int(np.random.uniform(0.0, T))
        t0 = random.randint(0, len_spectro - t)
        cloned[:, t0:t0+t] = 0
    return cloned

print("Подготовка данных...")

files_by_actor = defaultdict(list)
all_actors_ids = []

for actor_path in sorted(DATA_DIR.iterdir()):
    if actor_path.is_dir():
        actor_id = actor_path.name
        if actor_id not in all_actors_ids:
            all_actors_ids.append(actor_id)
        
        for file in actor_path.iterdir():
            if file.suffix == ".wav":
                files_by_actor[actor_id].append(file)

print(f"Найдено {len(all_actors_ids)} дикторов: {all_actors_ids}")


random.seed(42)
random.shuffle(all_actors_ids)

train_split = 0.7
val_split = 0.15

num_actors = len(all_actors_ids)
train_actors_count = int(num_actors * train_split)
val_actors_count = int(num_actors * val_split)

train_actors = all_actors_ids[:train_actors_count]
val_actors = all_actors_ids[train_actors_count : train_actors_count + val_actors_count]
test_actors = all_actors_ids[train_actors_count + val_actors_count:]

print(f"\nТренировочные дикторы ({len(train_actors)}): {train_actors}")
print(f"Валидационные дикторы ({len(val_actors)}): {val_actors}")
print(f"Тестовые дикторы ({len(test_actors)}): {test_actors}")


def collect_files(actor_list, file_dict):
    file_list = []
    for actor_id in actor_list:
        file_list.extend(file_dict[actor_id])
    return file_list

train_files = collect_files(train_actors, files_by_actor)
val_files = collect_files(val_actors, files_by_actor)
test_files = collect_files(test_actors, files_by_actor)

X_train, y_train = [], []
X_val, y_val = [], []
X_test, y_test = [], []

print(f"\n--- Обработка тренировочного набора ({len(train_files)} файлов) ---")
for file in train_files:
    filename = file.stem
    parts = filename.split('-')
    emotion = EMOTIONS[parts[2]]
    emotion_label = EMOTIONS_TO_NUM[emotion]
    audio, sr = librosa.load(str(file), sr=22050)
    
    # Оригинал + SpecAugment
    features_original = exctract_features(audio, sr)
    features_original_aug = time_masking(frequency_masking(features_original))
    X_train.append(features_original_aug)
    y_train.append(emotion_label)

    # Аугментация... (1x)
    for _ in range(2): # Добавляем 2 аугментированных копии
        choice = random.choice(['noise', 'pitch', 'stretch', 'shift'])
        if choice == 'noise':
            aug_audio = add_noise(audio)
        elif choice == 'pitch':
            aug_audio = pitch_shift(audio, sr, n_steps=random.uniform(-2, 2))
        elif choice == 'stretch':
            rate = random.uniform(0.8, 1.2)
            aug_audio = time_stretch(audio, rate=rate)
        else: # shift
            aug_audio = time_shift(audio)
        
        features_aug = exctract_features(aug_audio, sr)
        features_aug_spec = time_masking(frequency_masking(features_aug))

        X_train.append(features_aug_spec)
        y_train.append(emotion_label)

print(f"\n--- Обработка валидационного набора ({len(val_files)} файлов) ---")
for file in val_files:
    filename = file.stem
    parts = filename.split('-')
    emotion = EMOTIONS[parts[2]]
    emotion_label = EMOTIONS_TO_NUM[emotion]
    audio, sr = librosa.load(str(file), sr=22050)
    features_original = exctract_features(audio, sr)
    X_val.append(features_original)
    y_val.append(emotion_label)

print(f"\n--- Обработка тестового набора ({len(test_files)} файлов) ---")
for file in test_files:
    filename = file.stem
    parts = filename.split('-')
    emotion = EMOTIONS[parts[2]]
    emotion_label = EMOTIONS_TO_NUM[emotion]
    audio, sr = librosa.load(str(file), sr=22050)
    features_original = exctract_features(audio, sr)
    X_test.append(features_original)
    y_test.append(emotion_label)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)
X_test = np.array(X_test)
y_test = np.array(y_test)

shuffle_indices = np.random.permutation(len(X_train))
X_train = X_train[shuffle_indices]
y_train = y_train[shuffle_indices]

np.save(os.path.join(OUTPUT_DIR, "train_features.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "train_labels.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "val_features.npy"), X_val)
np.save(os.path.join(OUTPUT_DIR, "val_labels.npy"), y_val)
np.save(os.path.join(OUTPUT_DIR, "test_labels.npy"), y_test)
np.save(os.path.join(OUTPUT_DIR, "test_features.npy"), X_test)


print(f"\nСобрано {len(train_files) + len(val_files) + len(test_files)} исходных файлов.")
print(f"Всего признаков после аугментации: {len(X_train) + len(X_val) + len(X_test)}")
print(f"Тренировочный: {len(X_train)}, Валидационный: {len(X_val)}, Тестовый: {len(X_test)}")
print(f"Данные сохранены в папке {OUTPUT_DIR}")