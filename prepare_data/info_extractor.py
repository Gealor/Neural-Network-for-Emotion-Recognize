from pathlib import Path
from typing import Protocol, Tuple


EMOTIONS_TO_NUM = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'angry': 3,
    'fearful': 4,
    'disgust': 5,
    'surprised': 6
}

class AbstractInfoExtractor(Protocol):
    def extract_info(self, file: Path) -> Tuple[str, int]:
        ...


class RAVDESSExtractor(AbstractInfoExtractor):
    def __init__(self):
        self.EMOTIONS = {
            '01': 'neutral',
            '02': 'neutral', # 'calm'
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }

    def extract_info(self, file: Path) -> Tuple[str, int]:
        filename = file.stem
        parts = filename.split('-')

        actor_id = f"ravdess_{parts[6]}"

        emotion_mark = parts[2]
        emotion = self.EMOTIONS[emotion_mark]
        emotion_label = EMOTIONS_TO_NUM[emotion]
        return actor_id, emotion_label


class TESSExtractor(AbstractInfoExtractor):
    def __init__(self):
        self.EMOTIONS = {
            'neutral': 'neutral',
            'happy': 'happy',
            'sad': 'sad',
            'angry': 'angry',
            'fear': 'fearful',
            'disgust': 'disgust',
            'ps': 'surprised' # ps = pleasant surprise
        }

    def extract_info(self, file: Path) -> Tuple[str, int]:
        filename = file.stem
        parts = filename.split("_")

        actor_id = f"tess_{parts[0]}"

        emotion_mark = parts[2]
        emotion = self.EMOTIONS[emotion_mark]
        emotion_label = EMOTIONS_TO_NUM[emotion]

        return actor_id, emotion_label
    

class CREMADExtractor(AbstractInfoExtractor):
    def __init__(self):
        self.EMOTIONS = {
            'ANG': 'angry',
            'DIS': 'disgust',
            'FEA': 'fearful',
            'HAP': 'happy',
            'NEU': 'neutral',
            'SAD': 'sad'
        }
        
    def extract_info(self, file: Path) -> Tuple[str, int]:
        filename = file.stem
        parts = filename.split('_')

        actor_id = f"cremad_{parts[0]}"

        emotion_mark = parts[2]
        emotion = self.EMOTIONS[emotion_mark]
        emotion_label = EMOTIONS_TO_NUM[emotion]

        return actor_id, emotion_label