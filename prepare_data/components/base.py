from pathlib import Path
from typing import Generator, Protocol, Tuple

import numpy as np

type FileExtensions = Tuple[str, ...]

class MediaPipeline(Protocol):
    file_extensions: FileExtensions

    def process(self, file: Path, augment: bool = True) -> Generator[np.ndarray]:
        '''Обработка (и аугментация) одного файла'''
        ...