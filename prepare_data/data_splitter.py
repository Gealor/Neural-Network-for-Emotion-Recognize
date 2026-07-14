import random
from typing import Any, List, Tuple

from domain_models import SplitConfig

def split_data(
    data: List[Any], 
    split_config: SplitConfig,
    rng: random.Random,
    shuffle: bool = True
) -> Tuple[List[str], List[str], List[str]]:
    '''Разбивка данных на тренировочную, тестовую и валидационную выборки'''
    if shuffle:
        shuffled = data.copy()
        rng.shuffle(shuffled)
        actor_ids = shuffled

    num_actors = len(actor_ids)
    train_actors_count = int(num_actors * split_config.train)
    val_actors_count = int(num_actors * split_config.val)

    train_actors = actor_ids[:train_actors_count]
    val_actors = actor_ids[train_actors_count : train_actors_count + val_actors_count]
    test_actors = actor_ids[train_actors_count + val_actors_count:]

    print(f"\nТренировочные дикторы ({len(train_actors)}): {train_actors}\n"
        f"Валидационные дикторы ({len(val_actors)}): {val_actors}\n"
        f"Тестовые дикторы ({len(test_actors)}): {test_actors}"
    )

    return train_actors, val_actors, test_actors