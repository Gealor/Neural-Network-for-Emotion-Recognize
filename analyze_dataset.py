from pathlib import Path
from typing import List, Literal
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

import pandas as pd

from config import EMOTIONS, ROOT_DATA_DIR
from prepare_data.info_extractor import AbstractInfoExtractor, CREMADExtractor, RAVDESSExtractor, TESSExtractor

name_to_extractor = {
    "RAVDESS": RAVDESSExtractor(),
    "TESS": TESSExtractor(),
    "CREMA-D": CREMADExtractor()
}

def found_files(dataset_dir: Path) -> List[Path]:
    return list(dataset_dir.rglob("*.wav")) # rglob - рекурсивный поиск

def get_info_from_dataset(dataset_name: Literal["RAVDESS", "TESS", "CREMA-D"]):
    dataset_dir = ROOT_DATA_DIR / dataset_name

    extractor: AbstractInfoExtractor = name_to_extractor[dataset_name]

    audio_files = found_files(dataset_dir=dataset_dir)

    dataset_records = []

    for file_path in audio_files:
        try:
            actor_id, emotion_label = extractor.extract_info(file_path)
            actor_id, emotion_label = actor_id.split("_")[1], f"{emotion_label} ({EMOTIONS[emotion_label]})"
            gender = extractor.extract_gender(file_path)

            dataset_records.append({
                "dataset": dataset_name,
                "actor_id": actor_id,
                "emotion_label": emotion_label,
                "gender": gender,
                "file_name": file_path.stem,
                "file_path": str(file_path),
            })
        except (ValueError, KeyError, IndexError) as e:
            print(f"Ошибка в обработке файла {file_path}: {e}. Пропуск...")
            continue
    
    df = pd.DataFrame(dataset_records)
    print(f"[{dataset_name}] В таблицу добавлено {len(df)} файлов.")
    return df

def visualize_graph(final_df: pd.DataFrame):
    sns.set_theme(style="whitegrid", palette="muted")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # Разбивка по Датасетам
    dataset_counts = final_df.groupby(['emotion_label', 'dataset']).size().reset_index(name='count')
    dataset_pivot = dataset_counts.pivot(index='emotion_label', columns='dataset', values='count')
    dataset_pivot.plot(kind='bar', stacked=True, ax=axes[0], colormap='viridis', edgecolor='black')

    axes[0].set_title('Распределение классов по датасетам', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Класс эмоции', fontsize=12)
    axes[0].set_ylabel('Количество аудиофайлов', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend(title='Датасет', loc="lower right")

    # Разбивка по Полу
    gender_counts = final_df.groupby(['emotion_label', 'gender']).size().reset_index(name='count')
    gender_pivot = gender_counts.pivot(index='emotion_label', columns='gender', values='count')
    gender_pivot.plot(kind='bar', stacked=True, ax=axes[1], colormap='coolwarm', edgecolor='black')

    axes[1].set_title('Гендерный баланс внутри классов', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Класс эмоции', fontsize=12)
    axes[1].set_ylabel('Количество аудиофайлов', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend(title='Пол диктора', loc="lower right")

    # Общий заголовок и компановка
    plt.suptitle('Анализ баланса классов объединенного набора данных', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()

    plt.savefig("class_balance_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def extract_mel_energy(file_path: str) -> float:
    """Извлекает среднюю энергию низких частот (1-20 Мел-полос) из спектрограммы."""
    try:
        y, sr = librosa.load(file_path, sr=16000, duration=3.0)
        y, _ = librosa.effects.trim(y, top_db=25)

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Усреднение по оси времени, берем первые 20 мел-полос (нижние частоты)
        mean_mel_across_time = np.mean(mel_spec_db, axis=1) # для каждой строки вычисляется среднее
        low_freq_energy = np.mean(mean_mel_across_time[:20]) # анализируем только низкие частоты, чтобы анализировать больше полезной информации
        # т.к. в высоких частотах зачастую пустота (черные области)
        # low_freq_energy = np.mean(mean_mel_across_time) # анализируем всю спектрограмму
        return float(low_freq_energy)
    except Exception:
        return np.nan

def perform_anova(final_df: pd.DataFrame):
    print("\n=== Статистический анализ спектрограмм (ANOVA) ===")
    
    # Cлучайная подвыборку по 150 файлов каждого класса, чтобы сэкономить время. 
    print("Формирование выборки и извлечение признаков (пожалуйста, подождите)...")
    sampled_df = final_df.groupby('emotion_label').sample(n=150, random_state=42)
    # sampled_df = final_df.copy() # Раскомментировать для обработки всего датасета

    # Применяем функцию извлечения энергии к колонке file_path
    sampled_df['low_freq_energy'] = sampled_df['file_path'].apply(extract_mel_energy)
    sampled_df = sampled_df.dropna(subset=['low_freq_energy'])

    print("\n--- Точные значения энергии по классам (дБ) ---")
    # Группируем данные и считаем среднее (mean) и медиану (median)
    summary_stats = sampled_df.groupby('emotion_label')['low_freq_energy'].agg(['mean', 'median']).reset_index()
    
    for index, row in summary_stats.iterrows():
        print(f"{row['emotion_label']:20} | Среднее: {row['mean']:.2f} дБ | Медиана: {row['median']:.2f} дБ")
    print("-----------------------------------------------\n")

    # Подготавливаем списки значений для каждой группы (эмоции)
    groups = [group['low_freq_energy'].values for name, group in sampled_df.groupby('emotion_label')]

    # Проводим One-Way ANOVA (F-тест)
    f_statistic, p_value = stats.f_oneway(*groups)

    print(f"F-статистика: {f_statistic:.2f}")
    print(f"P-значение (p-value): {p_value:.5e}")

    if p_value < 0.05:
        print("ВЫВОД: Отвергаем H0. Фактор 'Эмоция' СТАТИСТИЧЕСКИ ЗНАЧИМО влияет на энергию спектрограммы.")
    else:
        print("ВЫВОД: Нет оснований отвергнуть H0.")

    # Рисуем Boxplot (Ящик с усами)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='emotion_label', y='low_freq_energy', data=sampled_df, palette='Set2', hue='emotion_label', legend=False)
    plt.title('Распределение энергии низких частот Мел-спектрограммы по эмоциям', fontsize=14, fontweight='bold')
    plt.ylabel('Средняя энергия в дБ (Low Freq Mels)', fontsize=12)
    plt.xlabel('Класс эмоции', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Сохраняем график
    plt.savefig("anova_spectrogram_boxplot.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    datasets = name_to_extractor.keys()

    all_dfs = []
    for ds in datasets:
        df = get_info_from_dataset(dataset_name=ds)
        all_dfs.append(df)

    final_df = pd.concat(all_dfs, ignore_index=True)

    print(f"\nВсего собрано аудиофайлов для анализа: {len(final_df)}")

    pivot_table = pd.crosstab(
        index=final_df['emotion_label'], 
        columns=[final_df['dataset'], final_df['gender']], 
        margins=True,
        margins_name="ИТОГО"
    )

    print("\nСводная таблица распределения классов:")
    print(pivot_table)
    final_df.to_csv("dataset_analysis.csv", index=False)
    pivot_table.to_csv("dataset_statistics.csv", index=False)
    visualize_graph(final_df)

    perform_anova(final_df)

if __name__=="__main__":
    main()

    # # Укажите путь к любому аудиофайлу из вашего датасета
    # file_path = Path(__file__).parent / "dataset" / "RAVDESS" / "Actor_01" / "03-01-04-01-01-01-01.wav"

    # # 1. Загрузка и создание ПОЛНОЙ матрицы (128 полос)
    # y, sr = librosa.load(file_path, sr=16000)
    # mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    # mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # # 2. Создание ОБРЕЗАННОЙ матрицы (Срез [:20] по оси частот)
    # # Мы берем с 0 по 19 строку, и все столбцы (время)
    # mel_spec_db_cropped = mel_spec_db[:20, :] 

    # # ==========================================
    # # ВИЗУАЛИЗАЦИЯ (СРАВНЕНИЕ)
    # # ==========================================
    # fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # # График 1: ПОЛНАЯ спектрограмма
    # # y_axis='mel' заставляет librosa нарисовать шкалу в Герцах (Hz)
    # img1 = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, ax=axes[0], cmap='magma')
    # axes[0].set_title('Оригинал: Полная Мел-спектрограмма (Массив из 128 строк)')
    # fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')

    # # График 2: ОБРЕЗАННАЯ спектрограмма
    # # Здесь мы специально не включаем перевод в Герцы, 
    # # чтобы вы увидели реальные ИНДЕКСЫ массива (от 0 до 19)
    # img2 = librosa.display.specshow(mel_spec_db_cropped, x_axis='time', ax=axes[1], cmap='magma')
    # axes[1].set_title('Срез [:20]: Только первые 20 элементов массива (Индексы 0-19)')
    # axes[1].set_ylabel('Индекс массива')
    # fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')

    # plt.tight_layout()
    # plt.show()
