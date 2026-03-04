import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils import compute_class_weight
import tensorflow as tf
from keras import models, layers, regularizers, optimizers, losses
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns

import config
from prepare_data.data_generator import DataGenerator

tf.config.optimizer.set_experimental_options({'layout_optimizer': False})


INPUT_SHAPE = (
    config.HEIGHT,
    config.WIDTH, 
    3 if config.INCLUDE_DELTAS else 1
)

# old CRNN model
def build_model(num_classes, input_shape = (config.HEIGHT, config.WIDTH, 1)):

    inputs = layers.Input(shape=input_shape)

    # Сверточные блоки
    x = layers.Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    _, h, w, c = x.shape
    new_shape = (int(w), int(h * c))
    x = layers.Reshape(target_shape=new_shape)(x)

    # RNN 
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.3))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False, dropout=0.3))(x)
    
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

# new model
def build_model_functional(num_classes, input_shape=(config.HEIGHT, config.WIDTH, 3)):
    # Используем Functional API вместо Sequential
    inputs = layers.Input(shape=input_shape)

    x = layers.GaussianNoise(0.05)(inputs)
    
    # Сверточные блоки
    x = layers.Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = se_block(x) # для данных, где три канала: оригинал, первая и вторая производные, чтобы модель определяла какой слой свертки выбрать
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.2)(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = se_block(x) 
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.3)(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = se_block(x)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.SpatialDropout2D(0.3)(x)

    x = layers.Conv2D(256, (3, 3), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = se_block(x)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.SpatialDropout2D(0.3)(x)
    
    # Reshape
    _, h, w, c = x.shape
    x = layers.Reshape((int(w), int(h * c)))(x)
    
    x = layers.Dense(128, activation='elu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.3)(x)

    # RNN
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=0.3))(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True, dropout=0.3))(x)
    
    # Context-Aware Attention
    att_weights = layers.Dense(1, activation='tanh')(x)
    att_weights = layers.Softmax(axis=1)(att_weights)

    att_weights_transposed = layers.Permute((2, 1))(att_weights)
    x_att = layers.Dot(axes=(2, 1))([att_weights_transposed, x])
    x_att = layers.Flatten()(x_att)

    x_max = layers.GlobalMaxPooling1D()(x)
    
    x = layers.Concatenate()([x_att, x_max]) # Объединяем "взвешенное среднее" и "максимумы"
    
    # Классификатор
    x = layers.Dense(128, activation='elu', kernel_regularizer=regularizers.l2(0.001))(x)
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


batch_size = 32
train_generator = DataGenerator('processed_data/X_train.npy', 'processed_data/y_train.npy', batch_size=batch_size, shuffle=True, augment=True, time_mask=15, freq_mask=8)
val_generator = DataGenerator('processed_data/X_val.npy', 'processed_data/y_val.npy', batch_size=batch_size, shuffle=False)
test_generator = DataGenerator('processed_data/X_test.npy', 'processed_data/y_test.npy', batch_size=batch_size, shuffle=False)

y_train = np.load('processed_data/y_train.npy')
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: weight for i, weight in enumerate(class_weights)}
del y_train

model = build_model_functional(num_classes=len(config.EMOTIONS.keys()), input_shape=INPUT_SHAPE)
# model = build_model(num_classes=len(config.EMOTIONS.keys()), input_shape=INPUT_SHAPE)
# model = models.load_model("best_model.h5") # загрузить последнюю лучшую модель
model.summary()


early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=0.00001)
ckpt = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)    
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

y_test = np.load('processed_data/y_test.npy', mmap_mode='r')
print(f"Размер предсказаний (y_pred): {y_pred.shape}")
print(f"Размер истинных меток (y_test): {y_test.shape}")

# -----------------------------------------------------------------------
# ВИЗУАЛИЗАЦИЯ ДАННЫХ 
# -----------------------------------------------------------------------
# # График Cyclic Learning Rate
# plt.figure(figsize=(10, 6))
# plt.plot(clr.history['iterations'], clr.history['lr'])
# plt.title('Cyclic Learning Rate ("triangular2" mode)')
# plt.xlabel('Batch Iterations')
# plt.ylabel('Learning Rate')
# plt.grid(True)
# plt.savefig('cyclic_learning_rate.png')

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Точность модели на тестовых данных: {accuracy:.4f}")
print(f"F1-метрика: {f1:.4f}")
print("Полный отчет классификации:")
print(classification_report(y_test, y_pred, target_names=config.EMOTIONS.values(), zero_division=0))

# Матрица ошибок (confusion matrix)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=config.EMOTIONS.values(), yticklabels=config.EMOTIONS.values())
plt.ylabel('Истинные значения')
plt.xlabel('Предсказанные значения')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

# График точности на тренировочных и валидационных данных
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Тренировочная точность')
plt.plot(history.history['val_accuracy'], label='Валидационная точность')
plt.title('Точность модели на тренировочных и валидационных данных')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.savefig('accuracy.png')

# График потерь
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Тренировочные потери')
plt.plot(history.history['val_loss'], label='Валидационные потери')
plt.title('Потери модели на тренировочных и валидационных данных')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()
plt.savefig('loss.png')