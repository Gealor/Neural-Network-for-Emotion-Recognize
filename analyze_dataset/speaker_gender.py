"""Пол спикера по имени файла — нужен только для анализа датасета, не для
подготовки данных. Вынесено из пайплайна (был `*Extractor.extract_gender`).
"""
from functools import cache
from pathlib import Path
from typing import Callable, Literal

import pandas as pd

import config

Gender = Literal["female", "male"]


def ravdess_gender(file: Path) -> Gender:
    actor_id = int(file.stem.split("-")[6])
    return "female" if actor_id % 2 == 0 else "male"


def tess_gender(file: Path) -> Gender:
    return "female"


@cache
def _cremad_gender_map() -> dict[int, str]:
    df = pd.read_csv(config.CREMA_D_DICTORS_INFO)
    return df.set_index("ActorID")["Sex"].to_dict()


def cremad_gender(file: Path) -> Gender:
    actor_id = int(file.stem.split("_")[0])
    return _cremad_gender_map()[actor_id].lower()


SPEAKER_GENDER: dict[str, Callable[[Path], Gender]] = {
    "RAVDESS": ravdess_gender,
    "TESS": tess_gender,
    "CREMA-D": cremad_gender,
}
