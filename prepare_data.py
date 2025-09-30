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


def exctract_features(audio, sr, max_pad_len=200):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    # нормализация длины (обрезаю/дополняю нулями)
    if mfcc.shape[1]<max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc

print("Подготовка данных...")

files, all_labels_for_split = [], []
for actor_path in DATA_DIR.iterdir():
    if actor_path.is_dir():
        for file in actor_path.iterdir():
            if file.suffix==".wav":
                files.append(file)
                filename = file.stem
                parts = filename.split('-')
                emotion = EMOTIONS[parts[2]]
                all_labels_for_split.append(EMOTIONS_TO_NUM[emotion])

train_files, temp_files, _, temp_labels = train_test_split(
    files, all_labels_for_split, test_size=0.3, random_state=42, stratify=all_labels_for_split
)

val_files, test_files, _, _ = train_test_split(
    temp_files, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

X_train, y_train = [], []
X_val, y_val = [], []
X_test, y_test = [], []

print("\n--- Обработка тренировочного набора ---")
for file in train_files:
    filename = file.stem
    parts = filename.split('-')
    emotion = EMOTIONS[parts[2]]
    emotion_label = EMOTIONS_TO_NUM[emotion]

    audio, sr = librosa.load(file, sr=22050)
    
    # Оригинал
    mfcc = exctract_features(audio, sr)
    X_train.append(mfcc)
    y_train.append(emotion_label)
    print(f"Файл с именем {filename} обработан (оригинал)...")

    # Аугментация с шумом
    audio_noisy = add_noise(audio)
    mfcc_noise = exctract_features(audio_noisy, sr)
    X_train.append(mfcc_noise)
    y_train.append(emotion_label)

    # Аугментация со сдвигом тона
    audio_pitch = pitch_shift(audio, sr, n_steps=random.uniform(-2, 2))
    mfcc_pitch = exctract_features(audio_pitch, sr)
    X_train.append(mfcc_pitch)
    y_train.append(emotion_label)

    # Аугментация с растягиванием аудио
    rate = random.uniform(0.8, 1.2) # Используем более щадящий диапазон
    audio_stretch = time_stretch(audio, rate=rate)
    mfcc_stretch = exctract_features(audio_stretch, sr)
    X_train.append(mfcc_stretch)
    y_train.append(emotion_label)

    # Аугментация по временному сдвигу
    audio_shift = time_shift(audio)
    mfcc_shift = exctract_features(audio_shift, sr)
    X_train.append(mfcc_shift)
    y_train.append(emotion_label)
    
    print(f"...аугментирован.")

print("\n--- Обработка валидационного набора ---")
for file in val_files:
    filename = file.stem
    parts = filename.split('-')
    emotion = EMOTIONS[parts[2]]
    emotion_label = EMOTIONS_TO_NUM[emotion]
    audio, sr = librosa.load(file, sr=22050)
    mfcc = exctract_features(audio, sr)
    X_val.append(mfcc)
    y_val.append(emotion_label)
    print(f"Файл с именем {filename} обработан (оригинал)...")

print("\n--- Обработка тестового набора ---")
for file in test_files:
    filename = file.stem
    parts = filename.split('-')
    emotion = EMOTIONS[parts[2]]
    emotion_label = EMOTIONS_TO_NUM[emotion]
    audio, sr = librosa.load(file, sr=22050)
    mfcc = exctract_features(audio, sr)
    X_test.append(mfcc)
    y_test.append(emotion_label)
    print(f"Файл с именем {filename} обработан (оригинал)...")


X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)
X_test = np.array(X_test)
y_test = np.array(y_test)


np.save(os.path.join(OUTPUT_DIR, "train_features.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "train_labels.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "val_features.npy"), X_val)
np.save(os.path.join(OUTPUT_DIR, "val_labels.npy"), y_val)
np.save(os.path.join(OUTPUT_DIR, "test_features.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "test_labels.npy"), y_test)

print(f"\nСобрано {len(files)} исходных файлов.")
print(f"Всего признаков после аугментации: {len(X_train) + len(X_val) + len(X_test)}")
print(f"Тренировочный: {len(X_train)}, Валидационный: {len(X_val)}, Тестовый: {len(X_test)}")
print(f"Данные сохранены в папке {OUTPUT_DIR}")