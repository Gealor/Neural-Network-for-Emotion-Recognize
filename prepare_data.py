import os
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

def exctract_features(file_path: pathlib.Path, max_pad_len=200):
    audio, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    if mfcc.shape[1]<max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc

print("Подготовка данных...")

files = []

for actor_path in DATA_DIR.iterdir():
    if actor_path.is_dir():
        for file in actor_path.iterdir():
            if file.suffix==".wav":
                files.append(file)

features, labels = [], []

for file in files:
    filename = file.stem # имя без расширения
    parts = filename.split('-')

    emotion = EMOTIONS[parts[2]]
    emotion_label = EMOTIONS_TO_NUM[emotion]

    mfcc = exctract_features(file)

    features.append(mfcc)
    labels.append(emotion_label)
    print(f"Файл с именем {filename} обработан")

features = np.array(features)
labels = np.array(labels)

X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

np.save(os.path.join(OUTPUT_DIR, "train_features.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "train_labels.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "val_features.npy"), X_val)
np.save(os.path.join(OUTPUT_DIR, "val_labels.npy"), y_val)
np.save(os.path.join(OUTPUT_DIR, "test_features.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "test_labels.npy"), y_test)

print(f"Собрано {len(files)} файлов.")
print(f"Тренировочный: {len(X_train)}, Валидационный: {len(X_val)}, Тестовый: {len(X_test)}")
print(f"Данные сохранены в папке {OUTPUT_DIR}")