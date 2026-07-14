from pathlib import Path
from typing import Generator, Iterable, Protocol

import numpy as np

type FileExtensions = Iterable[str]

class MediaPipeline(Protocol):
    file_extensions: FileExtensions

    def process(self, file: Path, augment: bool = True) -> Generator[np.ndarray]:
        '''Обработка (и аугментация) одного файла'''
        ...