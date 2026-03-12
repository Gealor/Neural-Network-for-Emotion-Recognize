from pathlib import Path
from typing import List, Literal

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
                "file_name": file_path.stem
            })
        except (ValueError, KeyError, IndexError) as e:
            print(f"Ошибка в обработке файла {file_path}: {e}. Пропуск...")
            continue
    
    df = pd.DataFrame(dataset_records)
    print(f"[{dataset_name}] В таблицу добавлено {len(df)} файлов.")
    return df

def main():
    datasets = ["RAVDESS", "TESS", "CREMA-D"]

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

if __name__=="__main__":
    main()
