from dataclasses import dataclass
import random
from typing import Iterable, Literal

from prepare_data.pipelines.base import MediaPipeline

type DatasetType = Literal["train", "test", "val"]

@dataclass(frozen=True)
class SplitConfig:
    train: float
    val: float

    def __post_init__(self):
        if self.train < 0 or self.val < 0:
            raise ValueError(f"train and val must be non-negative, got train={self.train}, val={self.val}")
        if self.test < 0:
            raise ValueError(f"train + val must not exceed 1.0, got train={self.train}, val={self.val}")
        if self.train + self.val > 1.0:
            raise ValueError(
                f"train + val must not exceed 1.0, got {self.train + self.val:.4f}"
            )

    @property
    def test(self) -> float:
        return 1-(self.train + self.val)


@dataclass(frozen=True)
class PipelineConfig:
    pipeline: MediaPipeline
    rng: random.Random

    @property
    def file_extensions(self) -> set[str]:
        formats = self.pipeline.file_extensions
        if not isinstance(formats, Iterable):
            formats = (formats, )
        extensions = set(format_.lower() for format_ in formats)
        return extensions