# Распознавание эмоций по речи

Нейросетевое распознавание эмоций по аудиозаписи речи на английском языке.
Шесть классов: neutral, happy, sad, angry, fearful, disgust. Данные — корпуса
RAVDESS, TESS и CREMA-D; разбиение на train/val/test спикер-независимое.

Исследуются два подхода:

- **CRNN** — собственная ResNet-CRNN с вниманием на лог-мел-спектрограммах;
- **SSL-энкодеры** — замороженные `wavlm-base-plus`, `wavlm-large`,
  `hubert-large-ll60k` с обучаемой классификационной головой.

## Структура

```text
├── ser_core/            # общее ядро: загрузка аудио, лог-мел-признаки, метки эмоций
├── research/            # исследовательский комплекс — см. research/README.md
│   ├── prepare_data/    #   подготовка данных (python -m research.prepare_data)
│   ├── model_training.py, ssl_head_training.py, extract_ssl_features.py
│   ├── analysis/        #   анализ датасетов (ноутбук, графики)
│   └── tools/           #   скачивание датасетов, проверка GPU
├── dataset/             # корпуса (.wav скачиваются отдельно)
├── processed_data*/     # подготовленные данные (не в git)
├── results/             # графики, чекпойнты голов, логи прогонов
└── pyproject.toml       # пакет и закреплённые версии зависимостей
```

`ser_core` не зависит ни от `research`, ни от TF/torch: его будет использовать
и веб-сервис распознавания, чтобы предобработка совпадала с обучением.

## Установка

Нужен **Python ≥ 3.12** (проверено на 3.13.12). Все команды — из корня репозитория.

### 1. Окружение

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

Или conda: `conda create -n ser python=3.13 && conda activate ser`.

### 2. Зависимости и сам проект

```bash
pip install -e ".[research,gpu]"   # обучение на GPU (Linux / WSL2)
pip install -e ".[research]"       # без GPU для TensorFlow
pip install -e .                   # только ядро ser_core (numpy, librosa)
```

| Группа | Что ставит |
| --- | --- |
| — (базовая) | `numpy`, `librosa` — зависимости `ser_core` |
| `research` | TensorFlow/Keras (CRNN), PyTorch + transformers (SSL-энкодеры), scikit-learn, pandas, scipy, matplotlib, seaborn, npy-append-array, kagglehub |
| `gpu` | `tensorflow[and-cuda]` — CUDA-библиотеки для TF; PyTorch ставит CUDA сам |

Версии закреплены точно (`==`) в `pyproject.toml`: на них получены результаты
НИР. Флаг `-e` (editable) ставит проект ссылкой на исходники: правки кода
действуют сразу, а пакеты `research` и `ser_core` импортируются из любого
каталога, в том числе из Jupyter.

### 3. Проверка GPU

```bash
python -m research.tools.test_gpu
```

Должно вывести `Num GPUs Available: 1` для TensorFlow и `True` для PyTorch.
В нативной Windows TensorFlow не поддерживает CUDA, поэтому обучение на
видеокарте идёт через WSL2. Настройка WSL2 для машинного обучения:

- <https://www.youtube.com/watch?v=qOJ49nkU4rY&list=PLw05BvhWaBV3TapljkXgg4eLh8O89S_Lv>
- <https://gist.github.com/raulqf/2d5f2b33549e56a6bb7c9f52a7fd471c>

### 4. Датасеты

```bash
python -m research.tools.download_dataset
```

Скрипт скачивает корпуса с Kaggle и печатает пути к ним; содержимое нужно
перенести в `dataset/RAVDESS/`, `dataset/TESS/`, `dataset/CREMA-D/`. Ожидаемая
структура — в [research/README.md](research/README.md#1-датасеты). Описание
CREMA-D: <https://pmc.ncbi.nlm.nih.gov/articles/PMC4313618/>.

## Быстрый старт

```bash
# CRNN
python -m research.prepare_data
python -m research.model_training

# SSL-энкодер + голова
python -m research.prepare_data --ssl
python -m research.extract_ssl_features --model wavlm-large
python -m research.ssl_head_training     # энкодер задаётся SSL_MODEL в скрипте
```

Скрипты запускаются только из корня и только через `python -m`. Параметры,
форматы выходных файлов, кросс-валидация по разбиениям и то, где сохраняются
результаты, — в [research/README.md](research/README.md).

## Результаты

- `results/runs.csv` — журнал всех прогонов (accuracy, weighted/macro F1);
  не в git, ведётся локально;
- `results/{first,second,third}_iteration/` — графики базовых прогонов CRNN;
- `results/ssl/<model>/seed-NN/` — головы SSL-энкодеров и их графики по трём
  независимым разбиениям.
