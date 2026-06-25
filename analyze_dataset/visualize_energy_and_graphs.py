from pathlib import Path
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def test_audio(file_path: Path):
    print("Обработка аудиофайла...")
    
    # Загрузка и обрезка тишины (чтобы анализировать только речь)
    y, sr = librosa.load(file_path, sr=16000)
    y, _ = librosa.effects.trim(y, top_db=25)

    # Вычисление Мел-спектрограммы и энергии (строго как в ANOVA)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Усредняем по времени, берем срез [:20]
    mean_mel_across_time = np.mean(mel_spec_db, axis=1)
    low_freq_energy = float(np.mean(mean_mel_across_time[:20]))

    reference_medians = {
        "Грусть (Sad)": -28.08,
        "Нейтраль (Neutral)": -30.91,
        "Страх (Fearful)": -35.31,
        "Отвращение (Disgust)": -34.85,
        "Радость (Happy)": -37.14,
        "Злость (Angry)": -41.66
    }

    # Находим самую близкую эмоцию математически
    closest_emotion = min(reference_medians.keys(), key=lambda k: abs(reference_medians[k] - low_freq_energy))

    print("\n" + "="*50)
    print(f"ФАЙЛ: {file_path.name}")
    print(f"Извлеченная энергия басов ([:20]): {low_freq_energy:.2f} дБ")
    print("="*50)
    print("СРАВНЕНИЕ С БАЗОЙ ЗНАНИЙ (ДАТАСЕТОМ):")
    for em, val in reference_medians.items():
        marker = " <--- ВАШ ФАЙЛ БЛИЖЕ ВСЕГО СЮДА" if em == closest_emotion else ""
        print(f"  {em:22}: ~ {val:.1f} дБ {marker}")
    print("="*50 + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

    # --- График 1: спектрограмма ---
    img = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, ax=axes[0], cmap='magma')
    axes[0].set_title(f'Мел-спектрограмма аудиозаписи (Файл: {file_path.name})', fontsize=12, fontweight='bold')
    fig.colorbar(img, ax=axes[0], format='%+2.0f dB')

    # --- График 2: Линейка ANOVA ---
    axes[1].set_title('Сравнение энергии аудиофайла со статистикой датасета (ANOVA)', fontsize=12, fontweight='bold')
    
    # Рисуем горизонтальную ось (линейку)
    axes[1].hlines(1, -60, -15, colors='gray', linestyles='solid', alpha=0.5, linewidth=2)

    # Отмечаем эталонные значения синими точками
    for em, val in reference_medians.items():
        axes[1].plot(val, 1, 'bo', alpha=0.6, markersize=8)
        # Подписываем названия эмоций
        axes[1].text(val, 1.05, em.split()[1].replace('(', '').replace(')', ''), 
                     rotation=45, ha='left', va='bottom', fontsize=10, color='blue')

    # Отмечаем НАШ ТЕКУЩИЙ ФАЙЛ огромной красной точкой
    axes[1].plot(low_freq_energy, 1, 'ro', markersize=14, markeredgecolor='black', zorder=5)
    axes[1].text(low_freq_energy, 0.9, f'ТЕКУЩИЙ ФАЙЛ\n({low_freq_energy:.1f} дБ)',
                 ha='center', va='top', fontsize=11, color='red', fontweight='bold')

    # Наводим красоту на нижнем графике
    axes[1].set_xlim([-50, -20]) # Ограничиваем ось X разумными рамками
    axes[1].set_ylim([0.5, 1.5]) # Ограничиваем ось Y (чтобы линия была по центру)
    axes[1].set_yticks([])       # Убираем цифры с оси Y (они тут не нужны)
    axes[1].set_xlabel('Средняя энергия низких частот (дБ)', fontsize=11)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Грустный файл "03-01-04-01-01-01-01.wav" 
    # Злой файл "03-01-05-01-01-01-01.wav"
    test_file = Path(__file__).parent.parent / "dataset" / "RAVDESS" / "Actor_01" / "03-01-05-01-01-01-01.wav"
    
    # Запускаем нашу магию
    test_audio(test_file)