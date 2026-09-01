# Цель проекта

Написать собственную нейронную сеть для задач распознавания звуков.
В данной ветке я проверяю новую архитектуре CRNN.

# Информация о датасете CREMA-D

https://pmc.ncbi.nlm.nih.gov/articles/PMC4313618/

# Описание

В проекте представлено четыре основных python скрипта:
- download_dataset: для скачивания датасета RAVDRESS(https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio?resource=download)

- prepare_data: для подготовки данных из нескольких корпусов для обучения модели (с аугментацией)
- model_training: новая архитектура модели, сфокусированная на новой версии prepare_data
- test_gpu: для тестирования - поддерживается ли обучение на видеокарте

## Структура подготовки данных (`prepare_data/`)

- `prepare_data.py` — точка входа: `run(build_default_config())`
- `prepare_data/runner.py` — оркестрация (конфиг → сплит по корпусам → запись → mean/std)
- `prepare_data/corpora.py` — реестр корпусов: как из имени файла достать спикера и метку эмоции
- `prepare_data/splitting.py` — спикер-независимое разбиение корпуса на train/val/test
- `prepare_data/pipelines/audio/` — загрузка аудио, мел-спектрограммы, аугментации
- `prepare_data/writing.py` — потоковая запись фич в `processed_data/*.npy`
- `prepare_data/stats.py` — расчёт `mean.npy` / `std.npy` по каналам
- `domain_models.py` — датаклассы конфига (`PrepConfig`, `AudioConfig`, `SplitConfig`)

Для быстрого прогона на срезе данных выставьте `config.LIMIT_PER_CORPUS` (напр. `200`).


# Предварительные действия:

1. Создайте виртуальное окружение командой

```
python -m env .venv
```

2. Установите зависимости из файла requirements.txt

3. Выполните скрипт download_dataset.py и перенесите датасеты из указанных папок в dataset и соответствующую папку

4. Запустить подготовку данных скриптом prepare_data.py
5. После этого можно запускать model_training.py

# ЗАМЕЧАНИЯ

В нативной Windows системе отсутствует поддержка обучения на CUDA ядрах, для обучения на видеокарте я использую WSL2. Гайды о том как настроить WSL2 для машинного обучения на видеокарте: 
https://www.youtube.com/watch?v=qOJ49nkU4rY&list=PLw05BvhWaBV3TapljkXgg4eLh8O89S_Lv
https://gist.github.com/raulqf/2d5f2b33549e56a6bb7c9f52a7fd471c

# Изменения:

1. Смена архитектуры на ResNet-CRNN
2. Добавлен Attention-слой
3. Снижена регуляризация до 1e-4

# Результат:
Точность модели возросла в среднем до 69-70% (3 итерации), что является прекрасным результатом.
С результатами можно ознакомиться в папке results_resnet