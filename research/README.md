# Исследовательский комплекс — запуск

Подготовка данных, обучение CRNN и классификационных голов над SSL-энкодерами,
анализ датасетов. Общий с будущим веб-сервисом код (загрузка аудио, признаки,
метки эмоций) лежит в `ser_core/`.

> Идёт рефакторинг (`docs/research-package-refactor-plan.md`). Подготовка данных
> пока запускается корневыми `prepare_data.py` / `prepare_ssl_data.py`; в фазе 3
> они заменяются на `python -m research.prepare_data [--ssl]` — этот файл будет
> обновлён вместе с ними.

## Общие правила

- **Все команды запускаются из корня репозитория.** Скрипты из `research/` —
  только через `python -m research.<имя>`, не `python research/<имя>.py`:
  иначе не найдутся пакеты `research` и `ser_core`.
- Python ≥ 3.12 (в коде используется оператор `type`). Сейчас окружение — 3.13.12.
- Зависимости: `pip install -r requirements.txt` (TF/Keras для CRNN,
  PyTorch + transformers для SSL-энкодеров).
- Обучение на GPU — через WSL2; проверка: `python -m research.tools.test_gpu`.
- **Скрипты перезаписывают свои выходы** (`processed_data*/`, `best_model.keras`,
  графики в `results/`) без архивирования. После каждого прогона, который нужно
  сохранить, переносите артефакты в именованную папку вручную (см. ниже).

## 1. Датасеты

```bash
python -m research.tools.download_dataset
```

Скачивает RAVDESS, CREMA-D и TESS через `kagglehub` и печатает пути к ним.
Содержимое нужно перенести вручную:

```
dataset/
├── RAVDESS/    # Actor_01/ … Actor_24/
├── TESS/
└── CREMA-D/    # .wav + VideoDemographics.csv (уже в репозитории)
```

`.wav` хранятся в Git LFS. Отсутствующий корпус пропускается с сообщением.

## 2. Подготовка данных

### CRNN (лог-мел-спектрограммы)

```bash
python prepare_data.py
```

- Спикер-независимое разбиение каждого корпуса 70/15/15 (seed 42),
  лог-мел 128×128 с дельтами, 2 аугментированные копии для train.
- Пересоздаёт `processed_data/`: `X_{train,val,test}.npy`, `y_*.npy`,
  `groups_*.npy` (корпус каждой записи), `mean.npy`/`std.npy` (по каналам, по train).

### SSL (манифест путей)

```bash
python prepare_ssl_data.py
```

- То же разбиение, но без признаков: только пути к файлам и метки, без дубликатов
  от аугментации.
- Пересоздаёт `processed_data_ssl/` (**включая `embeddings/`**):
  `paths_*.npy` (пути относительно корня репозитория), `y_*.npy`, `groups_*.npy`.
- Seed разбиения задаётся в `prepare_ssl_data.py` (`build_default_config(42)`).

### Быстрый прогон на срезе

`LIMIT_PER_CORPUS` в `research/config.py`: `None` — все файлы, `N` —
детерминированный срез ~N файлов на корпус по всем спикерам (секунды вместо
часов). Используется для smoke-проверок; для настоящих экспериментов — `None`.

## 3. CRNN — обучение

```bash
python -m research.model_training
```

- Читает `processed_data/`, нормализует `(X − mean) / (std + 1e-6)`, SpecAugment
  на train. До 200 эпох с ранней остановкой, seed 42 (`SEED` в скрипте).
- Выходы:
  - `best_model.keras` — **в текущем каталоге** (поэтому запуск из корня);
  - `results/{accuracy,loss,confusion_matrix}.png`;
  - `results/logs/train_<время>.log` — полный вывод;
  - строка в `results/runs.csv` (accuracy, weighted/macro F1, эпохи).

## 4. SSL-энкодер + голова

Два шага: извлечь эмбеддинги замороженным энкодером, затем обучить на них голову.

### 4.1 Эмбеддинги

```bash
python -m research.extract_ssl_features --model wavlm-large
```

| Параметр | Значения | По умолчанию |
|---|---|---|
| `--model` | `wavlm-base-plus`, `wavlm-large`, `hubert-large-ll60k` | `wavlm-base-plus` |
| `--splits` | любые из `train val test` | все три |
| `--limit N` | первые N файлов каждого сплита (смоук-тест) | все |

- Аудио: 16 кГц, моно, обрезка тишины 35 дБ, **без** приведения к фиксированной длине.
  Эмбеддинг — среднее по времени последнего скрытого слоя.
- Веса скачиваются с Hugging Face при первом запуске (кэш `~/.cache/huggingface`).
- Выход: `processed_data_ssl/embeddings/<model>_<split>.npy`.

### 4.2 Голова

```bash
python -m research.ssl_head_training
```

- Энкодер и seed обучения задаются **в коде**: `SSL_MODEL` и `SEED` в начале
  `research/ssl_head_training.py`. Нужны эмбеддинги этого энкодера для всех трёх
  сплитов.
- Выходы:
  - `results/ssl/best_head_<model>.keras`, `results/ssl/*.png`;
  - `results/logs/train_ssl_<model>_<время>.log`;
  - строка в `results/runs.csv` с пометкой `ssl_head:<model>`.

### Кросс-валидация по разбиениям (как в НИР)

TESS содержит двух спикеров, поэтому точность на одном разбиении нестабильна.
Модели сравниваются на трёх независимых разбиениях (seed 41, 42, 43), основная
метрика — точность на CREMA-D. Для каждого seed:

1. выставить seed в `prepare_ssl_data.py` → `python prepare_ssl_data.py`;
2. `python -m research.extract_ssl_features --model <model>`;
3. выставить `SSL_MODEL` → `python -m research.ssl_head_training`;
4. перенести `results/ssl/*` в `results/ssl/<model>/seed-<NN>/`.

## 5. Анализ датасетов

`research/analysis/analyze_dataset.ipynb` и скрипты в `research/analysis/`.
Первая ячейка ноутбука сама находит корень репозитория и добавляет его в
`sys.path`, так что рабочий каталог Jupyter не важен. Скрипты запускаются из
корня: `python -m research.analysis.visualize_spec`. Графики и таблицы
сохраняются в `research/analysis/` (`config.ANALYZE_RESULTS_DIR`).

## Где что лежит

| Путь | Что | В git |
|---|---|---|
| `dataset/` | исходные корпуса | да (LFS) |
| `processed_data/` | признаки CRNN | нет |
| `processed_data_ssl/` | манифест SSL и эмбеддинги | нет |
| `best_model.keras` | последний чекпойнт CRNN | да |
| `results/{first,second,third}_iteration/` | графики базовых прогонов CRNN | да |
| `results/ssl/<model>/seed-NN/` | головы и графики SSL по разбиениям | да |
| `results/logs/` | логи прогонов | нет |
| `results/runs.csv` | журнал всех прогонов | нет |
| `research/analysis/*.png, *.csv` | результаты анализа датасетов | да |
