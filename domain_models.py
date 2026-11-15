from dataclasses import dataclass
from typing import Literal

from ser_core.audio_config import AudioConfig

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
class PrepConfig:
    split: SplitConfig
    audio: AudioConfig
    seed: int = 42
    limit_per_corpus: int | None = None
