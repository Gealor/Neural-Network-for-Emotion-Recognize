from pathlib import Path

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Конфигурация (как у тебя)
n_fft = 2048
hop_length = 512
n_mels = 128  # config.HEIGHT

# 1. Загрузка и удаление тишины
file_path = Path(__file__).parent / "dataset" / "RAVDESS" / "Actor_01" / "03-01-04-01-01-01-01.wav"
audio, sr = librosa.load(file_path, sr=16000)
audio, _ = librosa.effects.trim(audio, top_db=25)

# --- ВАРИАНТ 1: Обычная (линейная) спектрограмма (STFT) ---
# Возвращает комплексные числа, берем модуль (np.abs)
stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
spectrogram = np.abs(stft)
# Переводим в децибелы для визуализации/модели
log_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

# --- ВАРИАНТ 2: Mel-спектрограмма (как у тебя) ---
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
# Переводим в децибелы (тут power_to_db, т.к. melspectrogram возвращает энергию)
log_mel_spectrogram = librosa.power_to_db(mel_spec, ref=np.max)

# --- ВИЗУАЛИЗАЦИЯ (чтобы увидеть разницу) ---
plt.figure(figsize=(12, 8))

# Рисуем обычную спектрограмму
plt.subplot(2, 1, 1)
librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Обычная (Линейная) Спектрограмма')

# Рисуем Mel-спектрограмму
plt.subplot(2, 1, 2)
librosa.display.specshow(log_mel_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-Спектрограмма')

plt.tight_layout()
plt.show()