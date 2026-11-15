# Исследовательский комплекс — запуск

Подготовка данных, обучение CRNN и классификационных голов над SSL-энкодерами,
анализ датасетов. Общий с будущим веб-сервисом код (загрузка аудио, признаки,
метки эмоций) лежит в `ser_core/`.

## Общие правила

- **Все команды запускаются из корня репозитория.** Скрипты из `research/` —
  только через `python -m research.<имя>`, не `python research/<имя>.py`:
  иначе не найдутся пакеты `research` и `ser_core`.
- Python ≥ 3.12 (в коде используется оператор `type`). Сейчас окружение — 3.13.12.
- Установка (из корня, в виртуальном окружении):

  ```bash
  pip install -e ".[research,gpu]"   # без GPU: ".[research]"
  ```

  Подробно об окружении и группах зависимостей — в корневом `README.md`.

  Версии закреплены в `pyproject.toml` — на них получены результаты НИР.
  Базовые зависимости (`numpy`, `librosa`) — это ядро `ser_core`; группа
  `research` добавляет TF/Keras (CRNN), PyTorch + transformers (SSL-энкодеры),
  scikit-learn, pandas, matplotlib и т. д. После `pip install -e` пакеты
  `research` и `ser_core` импортируются из любого каталога (в том числе из
  ноутбука), но **скрипты по-прежнему запускаются из корня**: `best_model.keras`
  пишется в текущий каталог.
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

`.wav` в git не хранятся (`*.wav` в `.gitignore`), их нужно скачать.
Отсутствующий корпус пропускается с сообщением.

## 2. Подготовка данных

### CRNN (лог-мел-спектрограммы)

```bash
python -m research.prepare_data [--seed 42]
```

- Спикер-независимое разбиение каждого корпуса 70/15/15 (`--seed`, по умолчанию 42),
  лог-мел 128×128 с дельтами, 2 аугментированные копии для train.
- Пересоздаёт `processed_data/`: `X_{train,val,test}.npy`, `y_*.npy`,
  `groups_*.npy` (корпус каждой записи), `mean.npy`/`std.npy` (по каналам, по train).

### SSL (манифест путей)

```bash
python -m research.prepare_data --ssl [--seed 42]
```

- То же разбиение, но без признаков: только пути к файлам и метки, без дубликатов
  от аугментации.
- Пересоздаёт `processed_data_ssl/` (**включая `embeddings/`**):
  `paths_*.npy` (пути относительно корня репозитория), `y_*.npy`, `groups_*.npy`.
- `--seed` задаёт разбиение на train/val/test. **При одном и том же seed разбиение
  не совпадает с CRNN-вариантом**: там один RNG расходуется и на разбиение, и на
  аугментацию, поэтому начиная со второго корпуса (порядок: RAVDESS, TESS,
  CREMA-D) спикеры распределяются иначе.

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

1. `python -m research.prepare_data --ssl --seed <NN>`;
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
| `dataset/` | исходные корпуса | только `VideoDemographics.csv` и заглушки, `.wav` — нет |
| `processed_data/` | признаки CRNN | нет |
| `processed_data_ssl/` | манифест SSL и эмбеддинги | нет |
| `best_model.keras` | последний чекпойнт CRNN | да |
| `results/{first,second,third}_iteration/` | графики базовых прогонов CRNN | да |
| `results/ssl/<model>/seed-NN/` | головы и графики SSL по разбиениям | да |
| `results/logs/` | логи прогонов | нет |
| `results/runs.csv` | журнал всех прогонов | нет |
| `research/analysis/*.png, *.csv` | результаты анализа датасетов | да |
