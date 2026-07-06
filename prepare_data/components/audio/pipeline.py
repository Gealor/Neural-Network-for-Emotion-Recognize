from pathlib import Path
import random
from typing import Callable, Dict, Generator, Tuple

import librosa
import numpy as np

from prepare_data.components.audio.feature_extraction import MelSpectrogramExtractor
from prepare_data.components.base import FileExtensions

type AudioAugmentFunction = Callable[[np.ndarray, int | float, random.Random], np.ndarray]

def add_noise(audio: np.ndarray, rng: random.Random, snr_db: int = 20) -> np.ndarray:
    """Добавляет шум на основе заданного соотношения сигнал/шум в дБ."""
    # Мощность сигнала
    audio_rms = np.sqrt(np.mean(audio**2))
    # Мощность шума на основе SNR
    snr_linear = 10**(snr_db / 20)
    noise_rms = audio_rms / snr_linear
    
    noise = np.array([rng.gauss(0, noise_rms) for _ in range(len(audio))])
    
    return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)

def pitch_shift(audio: np.ndarray, sr: int | float, n_steps: int | float = 2.0) -> np.ndarray:
    """Сдвигает высоту тона аудиосигнала."""
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

def time_stretch(audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """
    Растягивает или сжимает аудио по времени без изменения высоты тона.
    - rate > 1.0: ускоряет аудио (делает его короче)
    - rate < 1.0: замедляет аудио (делает его длиннее)
    """
    return librosa.effects.time_stretch(y=audio, rate=rate)

def time_shift(audio: np.ndarray, rng: random.Random, shift_max_ratio: float = 0.2) -> np.ndarray:
    """
    Сдвигает аудио во времени (циклически).
    Часть аудио, "выходящая" за один конец, появляется с другого.
    """
    shift_max = int(len(audio) * shift_max_ratio)
    shift_amount = rng.randint(-shift_max, shift_max)
    return np.roll(audio, shift_amount)

AUDIO_AUGMENT_REGISTER: Dict[str, AudioAugmentFunction] = {
    "noise": lambda audio, sr, rng: add_noise(audio, rng),
    "pitch": lambda audio, sr, rng: pitch_shift(audio, sr, n_steps=rng.uniform(-2, 2)),
    "stretch": lambda audio, sr, rng: time_stretch(audio, rate=rng.uniform(0.8, 1.2)),
    "shift": lambda audio, sr, rng: time_shift(audio, rng),
}

class AudioPipeline:
    file_extensions: FileExtensions = (".wav", )

    def __init__(self,
        feature_extractor: MelSpectrogramExtractor,
        augmenter_count: int = 2,
        sr: int = 16000,
        rng: random.Random | None = None,
    ):
        self.feature_extractor = feature_extractor
        self.augmenter_count = augmenter_count
        self.sr = sr
        self.rng = rng or random.Random()


    def _load(self, file: Path) -> Tuple[np.ndarray, int | float]:
        audio, sr = librosa.load(str(file), sr=self.sr)
        audio, _ = librosa.effects.trim(audio, top_db=35)
        return audio, sr


    def _augment_file(self, audio: np.ndarray, sr: int | float) -> Generator[np.ndarray]:
        '''Аугментирование данных, для расширения датасета.'''
        # Аугментация... (Nx)

        for _ in range(self.augmenter_count): # Добавляем N аугментированных копий
            choice = self.rng.choice(list(AUDIO_AUGMENT_REGISTER.keys()))
            augment = AUDIO_AUGMENT_REGISTER[choice]
            aug_audio = augment(audio, sr, self.rng)

            feature_aug = self.feature_extractor.extract(aug_audio, sr)
            yield feature_aug


    def process(self, file: Path, augment: bool = True) -> Generator[np.ndarray]:
        audio, sr = self._load(file)
        yield self.feature_extractor.extract(audio, sr)
        if augment:
            yield from self._augment_file(audio, sr)


def build_audio_pipeline(
    n_mels: int,
    max_pad_len: int,
    include_deltas: bool,
    rng: random.Random | None = None,
    augment_count: int = 2,
) -> AudioPipeline:
    extractor = MelSpectrogramExtractor(
        n_mels=n_mels,
        max_pad_len=max_pad_len,
        include_deltas=include_deltas,
    )
    return AudioPipeline(feature_extractor=extractor, augmenter_count=augment_count, rng=rng)


    