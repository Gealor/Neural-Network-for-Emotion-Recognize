import random
from typing import Any, List, Tuple


class ActorSplitter:
    def __init__(self, train_split: float, val_split: float, rng: random.Random | None = None):
        self.train_split = train_split
        self.val_split = val_split
        self.rng = rng or random.Random()

    def split(self, actor_ids: List[Any], shuffle: bool = True) -> Tuple[List[str], List[str], List[str]]:
        '''Разбивка данных на тренировочную, тестовую и валидационную выборки'''
        if shuffle:
            shuffled = actor_ids.copy()
            self.rng.shuffle(shuffled)
            actor_ids = shuffled

        num_actors = len(actor_ids)
        train_actors_count = int(num_actors * self.train_split)
        val_actors_count = int(num_actors * self.val_split)

        train_actors = actor_ids[:train_actors_count]
        val_actors = actor_ids[train_actors_count : train_actors_count + val_actors_count]
        test_actors = actor_ids[train_actors_count + val_actors_count:]

        print(f"\nТренировочные дикторы ({len(train_actors)}): {train_actors}\n"
            f"Валидационные дикторы ({len(val_actors)}): {val_actors}\n"
            f"Тестовые дикторы ({len(test_actors)}): {test_actors}"
        )

        return train_actors, val_actors, test_actors