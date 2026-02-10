import librosa
import numpy as np

import config


def add_noise(audio, snr_db=20):
    """Добавляет шум на основе заданного соотношения сигнал/шум в дБ."""
    # Мощность сигнала
    audio_rms = np.sqrt(np.mean(audio**2))
    # Мощность шума на основе SNR
    snr_linear = 10**(snr_db / 20)
    noise_rms = audio_rms / snr_linear
    
    rng = np.random.default_rng()
    noise = rng.standard_normal(size=audio.shape) * noise_rms
    
    return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)

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


def extract_features(audio, sr, n_mels=config.HEIGHT, max_pad_len=config.WIDTH):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=n_mels)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    # нормализация длины (обрезаю/дополняю нулями)
    if log_mel_spectrogram.shape[1] < max_pad_len:
        pad_width = max_pad_len - log_mel_spectrogram.shape[1]
        log_mel_spectrogram = np.pad(log_mel_spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        log_mel_spectrogram = log_mel_spectrogram[:, :max_pad_len]
    return log_mel_spectrogram