import random
from dataclasses import dataclass
from typing import Iterable, Literal

from prepare_data.pipelines.base import MediaPipeline

type DatasetType = Literal["train", "test", "val"]

@dataclass(frozen=True)
class SplitConfig:
    train: float
    val: float

    def __post_init__(self):
        if self.train < 0 or self.val < 0 or self.test < 0:
            raise ValueError(
                f"train/val/test must be non-negative and sum to 1.0, got "
                f"train={self.train}, val={self.val}, test={self.test:.4f}"
            )

    @property
    def test(self) -> float:
        return 1 - (self.train + self.val)


@dataclass(frozen=True)
class AudioConfig:
    sr: int = 16000
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    max_pad_len: int = 128
    include_deltas: bool = True
    trim_top_db: int = 35
    augment_count: int = 2


@dataclass(frozen=True)
class PrepConfig:
    split: SplitConfig
    audio: AudioConfig
    seed: int = 42
    limit_per_corpus: int | None = None


@dataclass(frozen=True)
class PipelineConfig:
    pipeline: MediaPipeline
    rng: random.Random

    @property
    def file_extensions(self) -> set[str]:
        formats = self.pipeline.file_extensions
        if not isinstance(formats, Iterable):
            formats = (formats, )
        extensions = {format_.lower() for format_ in formats}
        return extensions
