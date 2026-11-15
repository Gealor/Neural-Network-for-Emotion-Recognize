"""Подготовка данных.

    python -m research.prepare_data              # признаки CRNN -> processed_data/
    python -m research.prepare_data --ssl        # манифест SSL -> processed_data_ssl/
    python -m research.prepare_data --ssl --seed 43
"""
import argparse

from research.prepare_data.runner import build_default_config, run, run_ssl


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m research.prepare_data",
        description="Спикер-независимое разбиение корпусов и подготовка данных для обучения.",
    )
    parser.add_argument(
        "--ssl",
        action="store_true",
        help="подготовить манифест путей для SSL-энкодеров вместо лог-мел-признаков CRNN",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed разбиения на train/val/test и аугментаций (по умолчанию 42)",
    )
    args = parser.parse_args()

    cfg = build_default_config(args.seed)
    if args.ssl:
        run_ssl(cfg)
    else:
        run(cfg)


if __name__ == "__main__":
    main()
