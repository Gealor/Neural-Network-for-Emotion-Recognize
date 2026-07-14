from pathlib import Path

import numpy as np


def calculate_norm_params(path: Path):
    print("Расчет mean и std по каналам...")
    X = np.load(path, mmap_mode='r')
    
    # Определяем количество каналов
    # X.shape обычно (N, H, W) или (N, H, W, C)
    if len(X.shape) == 3:
        num_channels = 1
    else:
        num_channels = X.shape[-1]

    # Инициализируем накопители как float64 для точности
    sums = np.zeros(num_channels, dtype=np.float64)
    sums_sq = np.zeros(num_channels, dtype=np.float64)
    count = 0
    
    batch_size = 500 # Считаем кусками по 500 файлов для скорости
    num_samples = len(X)

    for i in range(0, num_samples, batch_size):
        end = min(i + batch_size, num_samples)
        batch = X[i:end].astype(np.float64) # Берем кусок данных
        
        # Если каналов несколько, считаем по осям (N, H, W)
        if num_channels > 1:
            sums += np.sum(batch, axis=(0, 1, 2))
            sums_sq += np.sum(batch**2, axis=(0, 1, 2))
            # Количество элементов в одном канале куска
            count += (end - i) * X.shape[1] * X.shape[2]
        else:
            sums[0] += np.sum(batch)
            sums_sq[0] += np.sum(batch**2)
            count += batch.size

    # Финальные расчеты
    mean = sums / count
    # std = sqrt( E[X^2] - (E[X])^2 )
    std = np.sqrt((sums_sq / count) - (mean**2))

    # Сохраняем (приводим к float32 для компактности)
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    # Меняем форму для удобного вычитания в генераторе: (1, 1, 1, C)
    if num_channels > 1:
        mean = mean.reshape(1, 1, 1, num_channels)
        std = std.reshape(1, 1, 1, num_channels)
    else:
        # Для 1 канала или если данные 3D
        mean = mean.reshape(1, 1, 1)
        std = std.reshape(1, 1, 1)

    np.save('processed_data/mean.npy', mean)
    np.save('processed_data/std.npy', std)

    print("Расчет окончен.")
    print("Mean per channel:", mean.flatten())
    print("Std per channel:", std.flatten())