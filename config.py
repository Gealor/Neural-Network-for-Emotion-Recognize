from pathlib import Path


# размерность данных
HEIGHT = 128
WIDTH = 128

# соотношение выборок
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15

# пути к папкам
ROOT_PATH = Path(__file__).parent

ROOT_DATA_DIR = ROOT_PATH / "dataset"
CREMA_D_DICTORS_INFO = ROOT_DATA_DIR / "CREMA-D" / "VideoDemographics.csv"

OUTPUT_DIR_NAME = 'processed_data'
OUTPUT_DIR = ROOT_PATH / OUTPUT_DIR_NAME

ANALYZE_RESULTS_DIR = ROOT_PATH / "analyze_dataset"

RESULTS_DIR = ROOT_PATH / "results"

# подготовка данных
INCLUDE_DELTAS = True # включать ли в данные первые и вторые производные
BATCH_SIZE = 2000

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


