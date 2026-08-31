import random
from pathlib import Path
from typing import Iterator

import numpy as np

from domain_models import AudioConfig
from prepare_data.pipelines.audio.augment import AUDIO_AUGMENT_REGISTER, AUGMENT_NAMES
from prepare_data.pipelines.audio.features import extract_logmel
from prepare_data.pipelines.audio.io import load_audio


class AudioPipeline:
    """Аудиофайл -> лог-мел-спектрограмма (оригинал + N аугментированных копий)."""

    def __init__(self, cfg: AudioConfig, rng: random.Random | None = None):
        self.cfg = cfg
        self.rng = rng or random.Random()

    def _augment(self, audio: np.ndarray, sr: int | float) -> Iterator[np.ndarray]:
        for _ in range(self.cfg.augment_count):
            name = self.rng.choice(AUGMENT_NAMES)
            aug_audio = AUDIO_AUGMENT_REGISTER[name](audio, sr, self.rng)
            yield extract_logmel(aug_audio, sr, self.cfg)

    def process(self, file: Path, *, augment: bool) -> Iterator[np.ndarray]:
        audio, sr = load_audio(file, sr=self.cfg.sr, trim_top_db=self.cfg.trim_top_db)
        yield extract_logmel(audio, sr, self.cfg)
        if augment:
            yield from self._augment(audio, sr)
