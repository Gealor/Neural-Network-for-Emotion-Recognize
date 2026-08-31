import random
from typing import Callable, Dict

import librosa
import numpy as np

AudioAugmentFunction = Callable[[np.ndarray, int | float, random.Random], np.ndarray]


def add_noise(audio: np.ndarray, rng: random.Random, snr_db: int = 20) -> np.ndarray:
    """Добавляет гауссов шум с заданным отношением сигнал/шум (дБ)."""
    audio_rms = np.sqrt(np.mean(audio**2))
    snr_linear = 10 ** (snr_db / 20)
    noise_rms = audio_rms / snr_linear

    noise = np.array([rng.gauss(0, noise_rms) for _ in range(len(audio))])

    return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)


def pitch_shift(audio: np.ndarray, sr: int | float, n_steps: int | float = 2.0) -> np.ndarray:
    """Сдвигает высоту тона."""
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)


def time_stretch(audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """Растягивает/сжимает по времени без изменения высоты тона
    (rate > 1.0 — быстрее/короче, rate < 1.0 — медленнее/длиннее)."""
    return librosa.effects.time_stretch(y=audio, rate=rate)


def time_shift(audio: np.ndarray, rng: random.Random, shift_max_ratio: float = 0.2) -> np.ndarray:
    """Циклический сдвиг во времени: край, ушедший за границу, появляется с другой."""
    shift_max = int(len(audio) * shift_max_ratio)
    shift_amount = rng.randint(-shift_max, shift_max)
    return np.roll(audio, shift_amount)


AUDIO_AUGMENT_REGISTER: Dict[str, AudioAugmentFunction] = {
    "noise": lambda audio, sr, rng: add_noise(audio, rng),
    "pitch": lambda audio, sr, rng: pitch_shift(audio, sr, n_steps=rng.uniform(-2, 2)),
    "stretch": lambda audio, sr, rng: time_stretch(audio, rate=rng.uniform(0.8, 1.2)),
    "shift": lambda audio, sr, rng: time_shift(audio, rng),
}

AUGMENT_NAMES = list(AUDIO_AUGMENT_REGISTER)
