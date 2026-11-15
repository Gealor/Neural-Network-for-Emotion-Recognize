from pathlib import Path

import librosa
import numpy as np


def load_audio(file: Path, sr: float, trim_top_db: int) -> tuple[np.ndarray, float]:
    """Загружает аудио с ресемплингом до sr и обрезает тишину по краям."""
    audio, sr = librosa.load(str(file), sr=sr)
    audio, _ = librosa.effects.trim(audio, top_db=trim_top_db)
    return audio, sr
