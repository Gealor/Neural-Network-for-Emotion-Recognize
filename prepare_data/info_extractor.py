from pathlib import Path
from typing import Literal, Protocol, Tuple

import pandas as pd

from config import CREMA_D_DICTORS_INFO, EMOTIONS_TO_NUM

Gender = Literal["female", "male"]

class AbstractInfoExtractor(Protocol):
    def extract_info(self, file: Path) -> Tuple[str, int]:
        ...

    def extract_gender(self, file: Path) -> Gender:
        ...

# TODO: убрать из RAVDESS метку 02, характеризующую нейтральную эмоцию(не факт, что calm эквивалентно neutral)
class RAVDESSExtractor:
    def __init__(self):
        self.EMOTIONS = {
            '01': 'neutral',
            # '02': 'neutral', # 'calm'
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            # '08': 'surprised'
        }

    def extract_info(self, file: Path) -> Tuple[str, int]:
        filename = file.stem
        parts = filename.split('-')

        actor_id = f"ravdess_{parts[6]}"

        emotion_mark = parts[2]
        emotion = self.EMOTIONS[emotion_mark]
        emotion_label = EMOTIONS_TO_NUM[emotion]
        return actor_id, emotion_label

    def extract_gender(self, file: Path) -> Gender:
        filename = file.stem
        parts = filename.split("-")

        actor_id = int(parts[6])
        return "female" if actor_id%2==0 else "male"

class TESSExtractor:
    def __init__(self):
        self.EMOTIONS = {
            'neutral': 'neutral',
            'happy': 'happy',
            'sad': 'sad',
            'angry': 'angry',
            'fear': 'fearful',
            'disgust': 'disgust',
            # 'ps': 'surprised' # ps = pleasant surprise
        }

    def extract_info(self, file: Path) -> Tuple[str, int]:
        filename = file.stem
        parts = filename.split("_")

        actor_id = f"tess_{parts[0]}"

        emotion_mark = parts[2]
        emotion = self.EMOTIONS[emotion_mark]
        emotion_label = EMOTIONS_TO_NUM[emotion]

        return actor_id, emotion_label
    
    def extract_gender(self, file: Path) -> Gender:
        return "female"

class CREMADExtractor:
    def __init__(self, dictors_info_path: Path = CREMA_D_DICTORS_INFO):
        self.EMOTIONS = {
            'ANG': 'angry',
            'DIS': 'disgust',
            'FEA': 'fearful',
            'HAP': 'happy',
            'NEU': 'neutral',
            'SAD': 'sad'
        }

        df = pd.read_csv(dictors_info_path)
        self.gender_map = df.set_index("ActorID")["Sex"].to_dict()
        
    def extract_info(self, file: Path) -> Tuple[str, int]:
        filename = file.stem
        parts = filename.split('_')

        actor_id = f"cremad_{parts[0]}"

        emotion_mark = parts[2]
        emotion = self.EMOTIONS[emotion_mark]
        emotion_label = EMOTIONS_TO_NUM[emotion]

        return actor_id, emotion_label
    
    def extract_gender(self, file: Path) -> Gender:
        filename = file.stem
        parts = filename.split("_")

        actor_id = int(parts[0])

        gender = self.gender_map[actor_id] 
        return gender.lower()