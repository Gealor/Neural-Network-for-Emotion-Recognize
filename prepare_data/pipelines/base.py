from pathlib import Path
from typing import Iterator, Protocol

import numpy as np


class MediaPipeline(Protocol):
    """Модальный шов: превращает один файл в набор feature-массивов
    (оригинал + N аугментаций). Реализации: AudioPipeline, позже VideoPipeline."""

    def process(self, file: Path, *, augment: bool) -> Iterator[np.ndarray]:
        ...
