from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import config
from config import EMOTIONS_TO_NUM

SpeakerFn = Callable[[Path], str]
LabelFn = Callable[[Path], int]


@dataclass(frozen=True)
class CorpusReader:
    root: Path
    extract_speaker: SpeakerFn
    extract_label: LabelFn
    file_glob: str = "**/*.wav"


# --- RAVDESS ---
_RAVDESS_EMOTIONS = {
    "01": "neutral",
    # "02": calm — не эквивалентно neutral, исключено
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    # "08": surprised — исключено
}


def ravdess_speaker(file: Path) -> str:
    return f"ravdess_{file.stem.split('-')[6]}"


def ravdess_label(file: Path) -> int:
    emotion_mark = file.stem.split("-")[2]
    return EMOTIONS_TO_NUM[_RAVDESS_EMOTIONS[emotion_mark]]


# --- TESS ---
_TESS_EMOTIONS = {
    "neutral": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fear": "fearful",
    "disgust": "disgust",
    # "ps" (pleasant surprise) — исключено
}


def tess_speaker(file: Path) -> str:
    return f"tess_{file.stem.split('_')[0]}"


def tess_label(file: Path) -> int:
    emotion_mark = file.stem.split("_")[2]
    return EMOTIONS_TO_NUM[_TESS_EMOTIONS[emotion_mark]]


# --- CREMA-D ---
_CREMAD_EMOTIONS = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


def cremad_speaker(file: Path) -> str:
    return f"cremad_{file.stem.split('_')[0]}"


def cremad_label(file: Path) -> int:
    emotion_mark = file.stem.split("_")[2]
    return EMOTIONS_TO_NUM[_CREMAD_EMOTIONS[emotion_mark]]


CORPORA: dict[str, CorpusReader] = {
    "RAVDESS": CorpusReader(
        root=config.ROOT_DATA_DIR / "RAVDESS",
        extract_speaker=ravdess_speaker,
        extract_label=ravdess_label,
    ),
    "TESS": CorpusReader(
        root=config.ROOT_DATA_DIR / "TESS",
        extract_speaker=tess_speaker,
        extract_label=tess_label,
    ),
    "CREMA-D": CorpusReader(
        root=config.ROOT_DATA_DIR / "CREMA-D",
        extract_speaker=cremad_speaker,
        extract_label=cremad_label,
    ),
}
