import librosa
import numpy as np

# В случае чего, можно расширить протоколом, если будут другие имплементации извлечения
class MelSpectrogramExtractor:
    def __init__(self, n_mels: int, max_pad_len: int, include_deltas: bool, hop_length: int = 512, n_fft: int = 2048):
        self.n_mels = n_mels
        self.max_pad_len = max_pad_len
        self.include_deltas = include_deltas
        self.hop_length = hop_length
        self.n_fft = n_fft

    def _padding_audio(self, audio: np.ndarray, target_audio_len: int) -> np.ndarray:
        if len(audio) < target_audio_len:
            pad_total = target_audio_len - len(audio)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            padded_audio = np.pad(audio, (pad_left, pad_right), mode='constant', constant_values=0)
        else:
            start = (len(audio) - target_audio_len) // 2
            padded_audio = audio[start : start + target_audio_len]

        return padded_audio

    def _include_deltas(self, log_mel_spectrogram: np.ndarray) -> np.ndarray:
        '''Вычисляем первую и вторую производные'''
        delta = librosa.feature.delta(log_mel_spectrogram)
        delta2 = librosa.feature.delta(log_mel_spectrogram, order=2)
        
        # Склеиваем в 3 канала: (128, 128, 3)
        features = np.stack([log_mel_spectrogram, delta, delta2], axis=-1)
        return features

    def extract(self, audio: np.ndarray, sr: int | float) -> np.ndarray:
        '''Извлечение признаков (мел-спектрограммы)'''
        target_audio_len = self.max_pad_len * self.hop_length

        audio = self._padding_audio(audio, target_audio_len)

        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        log_mel_spectrogram = np.maximum(log_mel_spectrogram, -80.0)
        log_mel_spectrogram = log_mel_spectrogram[:, :self.max_pad_len]

        if self.include_deltas:
            log_mel_spectrogram = self._include_deltas(log_mel_spectrogram)
        
        return log_mel_spectrogram