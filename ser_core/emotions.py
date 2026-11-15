# Метки классификатора. Порядок индексов = порядок выходов модели:
# сервис по нему переводит номер выхода в название эмоции.
EMOTIONS_TO_NUM = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'angry': 3,
    'fearful': 4,
    'disgust': 5,
    # 'surprised': 6
}

EMOTIONS = {
    value: key
    for key, value in EMOTIONS_TO_NUM.items()
}
