from dataclasses import dataclass
import random
from typing import Iterable, Literal

from prepare_data.pipelines.base import MediaPipeline

EPS = float("1e-10")
type DatasetType = Literal["train", "test", "val"]

@dataclass(frozen=True)
class SplitConfig:
    train: float
    val: float

    def __post_init__(self):
        if (self.train + self.val + self.test) - 1.0 > EPS:
            raise ValueError("Summary splits values must be equal 1")

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