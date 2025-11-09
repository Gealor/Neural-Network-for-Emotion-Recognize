import tensorflow as tf
import numpy as np

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_path, y_path, mean, std, batch_size=32, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.x = np.load(x_path, mmap_mode='r')
        self.y = np.load(y_path, mmap_mode='r')
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.x))
        self.on_epoch_end()

    def __len__(self):
        # Количество батчей за эпоху
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        # Генерирует один батч данных
        # 1. Выбираем индексы для батча
        batch_indices = self.indices[index*self.batch_size:min((index+1)*self.batch_size, len(self.x))]
        
        # 2. Загружаем данные по этим индексам (mmap_mode делает это эффективно)
        X_batch = self.x[batch_indices]
        y_batch = self.y[batch_indices]
        
        # 3. Применяем предобработку "на лету"
        X_batch_norm = (X_batch - self.mean) / (self.std + 1e-6)
        X_batch_final = np.expand_dims(X_batch_norm, -1)
        
        return X_batch_final, y_batch

    def on_epoch_end(self):
        # Перемешиваем индексы в конце каждой эпохи
        if self.shuffle:
            np.random.shuffle(self.indices)