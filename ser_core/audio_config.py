from dataclasses import dataclass


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
