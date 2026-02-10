import numpy as np

def calculate():
    print("Расчет mean и std для нормализации...")
    X_train = np.load('processed_data/X_train.npy', mmap_mode='r')

    # Расчет mean и std по оси частот (axis=1 в 2D представлении)
    mean = np.mean(X_train, axis=(0, 2), keepdims=True)
    std = np.std(X_train, axis=(0, 2), keepdims=True)

    np.save('processed_data/mean.npy', mean)
    np.save('processed_data/std.npy', std)

    print("Mean и std сохранены.")
    print("Форма mean:", mean.shape)