from pathlib import Path
from typing import Iterable


def get_all_files_by_format(data_dir: Path, formats: Iterable[str]) -> list[Path]:
    """Ищет файлы поддерживаемых расширений"""
    print(f"Поддерживаемые расширения: {formats}")

    files = []
    for file_path in data_dir.glob('**/*'):
        if file_path.suffix.lower() not in formats:
            if file_path.is_file():
                print(f"Пропущен файл {file_path.name}. Неподдерживаемый формат: {file_path.suffix}")
            continue

        files.append(file_path)

    return files
