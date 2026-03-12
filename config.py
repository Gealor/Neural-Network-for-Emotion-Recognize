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
CREMA_D_DICTORS_INFO = ROOT_DATA_DIR / "CREMA-D" / "VideoDemographics.csv"

# подготовка данных
INCLUDE_DELTAS = False # включать ли в данные первые и вторые производные

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
    value: key 
    for key, value in EMOTIONS_TO_NUM.items()
}


