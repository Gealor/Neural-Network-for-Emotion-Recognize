import random
import librosa
import numpy as np


def add_noise(audio, noise_factor=0.005):
    """Добавляет случайный Гауссовский шум."""
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio.astype(type(audio[0]))

def pitch_shift(audio, sr, n_steps=2.0):
    """Сдвигает высоту тона аудиосигнала."""
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

def time_stretch(audio, rate=1.0):
    """
    Растягивает или сжимает аудио по времени без изменения высоты тона.
    - rate > 1.0: ускоряет аудио (делает его короче)
    - rate < 1.0: замедляет аудио (делает его длиннее)
    """
    return librosa.effects.time_stretch(y=audio, rate=rate)

def time_shift(audio, shift_max_ratio=0.2):
    """
    Сдвигает аудио во времени (циклически).
    Часть аудио, "выходящая" за один конец, появляется с другого.
    """
    shift_max = int(len(audio) * shift_max_ratio)
    shift_amount = np.random.randint(-shift_max, shift_max)
    return np.roll(audio, shift_amount)


def extract_features(audio, sr, n_mels=128, max_pad_len=200):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=n_mels)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    # нормализация длины (обрезаю/дополняю нулями)
    if log_mel_spectrogram.shape[1] < max_pad_len:
        pad_width = max_pad_len - log_mel_spectrogram.shape[1]
        log_mel_spectrogram = np.pad(log_mel_spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        log_mel_spectrogram = log_mel_spectrogram[:, :max_pad_len]
    return log_mel_spectrogram

# SpecAugment, изменяет частотные полосы, чтобы менять тембр голоса и избежать привыкание модели к тембру
def frequency_masking(spectrogram, F=15, num_masks=1): # F - максимальная ширина маски
    cloned = spectrogram.copy()
    num_mel_channels = cloned.shape[0]
    for _ in range(num_masks):
        f = int(np.random.uniform(0.0, F))
        f0 = random.randint(0, num_mel_channels - f)
        cloned[f0:f0+f, :] = 0
    return cloned

def time_masking(spectrogram, T=25, num_masks=1): # T - максимальная ширина маски
    cloned = spectrogram.copy()
    len_spectro = cloned.shape[1]
    for _ in range(num_masks):
        t = int(np.random.uniform(0.0, T))
        t0 = random.randint(0, len_spectro - t)
        cloned[:, t0:t0+t] = 0
    return cloned