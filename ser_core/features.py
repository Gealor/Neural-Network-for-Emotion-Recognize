import librosa
import numpy as np

from domain_models import AudioConfig


def _pad_or_crop(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Доводит аудио до target_len: симметричный zero-pad или центральная обрезка."""
    if len(audio) < target_len:
        pad_total = target_len - len(audio)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(audio, (pad_left, pad_right), mode="constant", constant_values=0)

    start = (len(audio) - target_len) // 2
    return audio[start : start + target_len]


def _stack_deltas(log_mel: np.ndarray) -> np.ndarray:
    """Склеивает лог-мел с его первой и второй производными в 3 канала."""
    delta = librosa.feature.delta(log_mel)
    delta2 = librosa.feature.delta(log_mel, order=2)
    return np.stack([log_mel, delta, delta2], axis=-1)


def extract_logmel(audio: np.ndarray, sr: int | float, cfg: AudioConfig) -> np.ndarray:
    """Лог-мел-спектрограмма файла, опционально с дельта-каналами."""
    target_len = cfg.max_pad_len * cfg.hop_length
    audio = _pad_or_crop(audio, target_len)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = np.maximum(log_mel, -80.0)
    log_mel = log_mel[:, : cfg.max_pad_len]

    if cfg.include_deltas:
        log_mel = _stack_deltas(log_mel)

    return log_mel
