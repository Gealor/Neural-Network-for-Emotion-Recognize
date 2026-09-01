import sys
from datetime import datetime

import numpy as np
from sklearn.utils import compute_class_weight
import tensorflow as tf
from keras import models, layers, regularizers, optimizers, losses
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import config
from graphs_and_report import build_accuracy_graph, build_confusion_matrix, build_loss_graph, build_report
from prepare_data.data_generator import DataGenerator

tf.config.optimizer.set_experimental_options({'layout_optimizer': False})


class Tee:
    """Дублирует вывод в консоль и в файл.

    В консоль данные пишутся как есть (чтобы прогресс-бар model.fit
    нормально анимировался через \\r). В файл же попадают только
    завершённые строки: промежуточные перезаписи по \\r отбрасываются,
    и сохраняется лишь финальное состояние строки перед \\n — иначе
    прогресс-бар за каждую эпоху дублировался бы построчно на каждый шаг.
    """

    def __init__(self, console_stream, file_stream):
        self.console = console_stream
        self.file = file_stream
        self._file_line_buffer = ""

    def write(self, data):
        self.console.write(data)

        for char in data:
            if char == '\r':
                self._file_line_buffer = ""
            elif char == '\n':
                self.file.write(self._file_line_buffer + '\n')
                self._file_line_buffer = ""
            else:
                self._file_line_buffer += char

    def flush(self):
        self.console.flush()
        self.file.flush()


LOG_DIR = config.RESULTS_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_path = LOG_DIR / f"train_{datetime.now():%Y%m%d_%H%M%S}.log"
log_file = open(log_path, "w", encoding="utf-8")

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

print(f"Логирование вывода в файл: {log_path}")


INPUT_SHAPE = (
    config.HEIGHT,
    config.WIDTH, 
    3 if config.INCLUDE_DELTAS else 1
)

# CRNN model
def build_model(num_classes, input_shape = (config.HEIGHT, config.WIDTH, 1)):

    inputs = layers.Input(shape=input_shape)

    # Сверточные блоки
    x = layers.Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 1))(x) 
    x = layers.Dropout(0.3)(x)

    _, h, w, c = x.shape
    new_shape = (int(w), int(h * c))
    x = layers.Reshape(target_shape=new_shape)(x)

    # RNN 
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True, dropout=0.3))(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=False, dropout=0.3))(x)
    
    # Классификатор
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0003),
        loss=losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    return model


def se_block(inputs, ratio=8):
    """Squeeze-and-Excitation блок для внимания на каналы"""
    channel_axis = -1
    filters = inputs.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = layers.GlobalAveragePooling2D()(inputs)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', use_bias=False)(se)

    return layers.Multiply()([inputs, se])

# new model (ResNet)
def build_model_functional(num_classes, input_shape=(config.HEIGHT, config.WIDTH, 3)):
    # Используем Functional API вместо Sequential
    inputs = layers.Input(shape=input_shape)
    regular = regularizers.l2(0.001)

    x = layers.GaussianNoise(0.01)(inputs)

    def res_block(x_in, filters, pool_size, dropout_rate):
        # Shortcut (Короткий путь)
        if x_in.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)(x_in)
            shortcut = layers.BatchNormalization()(shortcut)
        else:
            shortcut = x_in

        # Основной путь (2 свертки)
        x_path = layers.Conv2D(filters, (3, 3), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x_in)
        x_path = layers.BatchNormalization()(x_path)
        x_path = layers.Activation('relu')(x_path)

        x_path = layers.Conv2D(filters, (3, 3), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x_path)
        x_path = layers.BatchNormalization()(x_path)

        # Сложение (Residual Connection)
        x_path = layers.Add()([shortcut, x_path])
        x_path = layers.Activation('relu')(x_path)
        
        # Пулинг и Дропаут
        x_path = layers.MaxPooling2D(pool_size=pool_size)(x_path)
        x_path = layers.Dropout(dropout_rate)(x_path)
        
        return x_path
    
    # Сверточные блоки
    x = layers.Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = res_block(x, filters=32, pool_size=(2, 2), dropout_rate=0.1)
    x = res_block(x, filters=64, pool_size=(2, 2), dropout_rate=0.1)
    x = res_block(x, filters=128, pool_size=(2, 1), dropout_rate=0.2)
    x = res_block(x, filters=128, pool_size=(2, 1), dropout_rate=0.2)

    # Reshape
    _, h, w, c = x.shape
    x = layers.Reshape((int(w), int(h * c)))(x)

    # RNN
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.3))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.3))(x)
    
    # Context-Aware Attention
    att_weights = layers.Dense(1, activation='tanh')(x)
    att_weights = layers.Softmax(axis=1)(att_weights)

    att_weights_transposed = layers.Permute((2, 1))(att_weights)
    x_att = layers.Dot(axes=(2, 1))([att_weights_transposed, x])
    x_att = layers.Flatten()(x_att)

    x_max = layers.GlobalMaxPooling1D()(x)
    
    x = layers.Concatenate()([x_att, x_max]) # Объединяем "взвешенное среднее" и "максимумы"

    # Классификатор
    x = layers.Dense(128, activation='relu', kernel_regularizer=regular)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0003, amsgrad=True),
        loss=losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    return model


batch_size = 32
train_generator = DataGenerator(config.OUTPUT_DIR / 'X_train.npy', config.OUTPUT_DIR / 'y_train.npy', batch_size=batch_size, shuffle=True, augment=True, time_mask=20, freq_mask=16)
val_generator = DataGenerator(config.OUTPUT_DIR / 'X_val.npy', config.OUTPUT_DIR / 'y_val.npy', batch_size=batch_size, shuffle=False)
test_generator = DataGenerator(config.OUTPUT_DIR / 'X_test.npy', config.OUTPUT_DIR / 'y_test.npy', batch_size=batch_size, shuffle=False)

y_train = np.load(config.OUTPUT_DIR / 'y_train.npy')
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: weight for i, weight in enumerate(class_weights)}
del y_train

model = build_model_functional(num_classes=len(config.EMOTIONS.keys()), input_shape=INPUT_SHAPE)
# model = build_model(num_classes=len(config.EMOTIONS.keys()), input_shape=INPUT_SHAPE)
# model = models.load_model("best_model.h5") # загрузить последнюю лучшую модель
model.summary()


early_stop = EarlyStopping(monitor='val_accuracy', mode='max', patience=8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', mode='max', factor=0.2, patience=6, min_lr=0.00001)
ckpt = ModelCheckpoint('best_model.keras', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
# Обучение модели
history = model.fit(
    train_generator,
    epochs=200,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[
        early_stop,
        reduce_lr,
        ckpt
    ],
)

print("\nОценка на тестовых данных...")
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=-1)

y_test = np.load(config.OUTPUT_DIR / 'y_test.npy', mmap_mode='r')
print(f"Размер предсказаний (y_pred): {y_pred.shape}")
print(f"Размер истинных меток (y_test): {y_test.shape}")


build_report(y_test, y_pred)

# Матрица ошибок (confusion matrix)
build_confusion_matrix(y_test, y_pred)

# График точности на тренировочных и валидационных данных
build_accuracy_graph(history)

# График потерь
build_loss_graph(history)
