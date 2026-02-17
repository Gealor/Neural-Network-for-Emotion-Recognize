from pathlib import Path


# размерность данных
HEIGHT = 128
WIDTH = 128

# соотношение выборок
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15

# пути к папкам
OUTPUT_DIR_NAME = 'processed_data'
ROOT_DATA_DIR = Path(__file__).parent / "dataset"
OUTPUT_DIR = Path(__file__).parent / OUTPUT_DIR_NAME


# метки для классификатора

EMOTIONS_TO_NUM = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'angry': 3,
    'fearful': 4,
    'disgust': 5,
    # 'surprised': 6
}

EMOTIONS = {
    0: 'neutral',
    1: 'happy',
    2: 'sad',
    3: 'angry',
    4: 'fearful',
    5: 'disgust',
    # 6: 'surprised'
}

