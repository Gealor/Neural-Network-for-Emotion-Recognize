import keras
import numpy as np

import config

class DataGenerator(keras.utils.Sequence):
    def __init__(
        self, 
        x_path, 
        y_path, 
        batch_size=32, 
        shuffle=True, 
        augment=False,
        time_mask=10,
        freq_mask=10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.x = np.load(x_path, mmap_mode='r')
        self.y = np.load(y_path, mmap_mode='r')
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.time_mask = time_mask
        self.freq_mask = freq_mask
        self.indices = np.arange(len(self.x))
        self.count_classes = len(config.EMOTIONS.keys())
        self.on_epoch_end()

    def __len__(self):
        # Количество батчей за эпоху
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        # Генерирует один батч данных
        # 1. Выбираем индексы для батча
        batch_indices = self.indices[index*self.batch_size:min((index+1)*self.batch_size, len(self.x))]
        
        # 2. Загружаем данные по этим индексам (mmap_mode делает это эффективно)
        X_batch = self.x[batch_indices].copy()
        y_batch = self.y[batch_indices]

        for i in range(len(X_batch)):
            if self.augment and np.random.rand() < 0.7:
                X_batch[i] = self.spec_augment(X_batch[i])
        
            sample = X_batch[i]
            # Z-score normalization по каналам
            for c in range(3):
                channel = sample[:, :, c]
                c_mean = np.mean(channel)
                c_std = np.std(channel) + 1e-6
                sample[:, :, c] = (channel - c_mean) / c_std
            X_batch[i] = sample

        X_batch_final = np.expand_dims(X_batch, -1)

        y_batch_one_hot = keras.utils.to_categorical(y_batch, num_classes=self.count_classes)
        
        return X_batch_final, y_batch_one_hot
    
    def spec_augment(self, mel_spec):
        augmented = mel_spec.copy()
        spec_mean = np.mean(augmented)
        
        # Frequency masking
        f = augmented.shape[0]
        f_mask = np.random.randint(1, self.freq_mask+1)
        f_start = np.random.randint(0, f - f_mask) if f > f_mask else 0
        if f_mask > 0 and f_start >= 0 and f_start + f_mask <= f:
            augmented[f_start:f_start + f_mask, :] = spec_mean
        
        # Time masking
        t = augmented.shape[1]
        t_mask = np.random.randint(1, self.time_mask+1)
        t_start = np.random.randint(0, t - t_mask) if t > t_mask else 0
        if t_mask > 0 and t_start >= 0 and t_start + t_mask <= t:
            augmented[:, t_start:t_start + t_mask] = spec_mean
        
        return augmented
    

    def on_epoch_end(self):
        # Перемешиваем индексы в конце каждой эпохи
        if self.shuffle:
            np.random.shuffle(self.indices)