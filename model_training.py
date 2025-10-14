import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils import compute_class_weight
import tensorflow as tf
from keras import models, layers, regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns


EMOTIONS = {
    0: 'neutral',
    1: 'calm',
    2: 'happy',
    3: 'sad',
    4: 'angry',
    5: 'fearful',
    6: 'disgust',
    7: 'surprised'
}

# CNN модель
def build_model(num_classes, input_shape = (128, 200, 1)):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Сверточные блоки
        layers.Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(256, (3, 3), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # Классификатор
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Загрузка подготовленных данных
X_train = np.load('processed_data/train_features.npy')
y_train = np.load('processed_data/train_labels.npy')
X_val = np.load('processed_data/val_features.npy')
y_val = np.load('processed_data/val_labels.npy')
X_test = np.load('processed_data/test_features.npy')
y_test = np.load('processed_data/test_labels.npy')

# ------------------------------------------------------------
# НОРМАЛИЗАЦИЯ ДАННЫХ
# ------------------------------------------------------------
X_train = np.expand_dims(X_train, -1)
X_val = np.expand_dims(X_val, -1)
X_test = np.expand_dims(X_test, -1)

n_samples_train, n_mfcc, n_frames, n_channels = X_train.shape
X_train_reshaped = X_train.reshape(n_samples_train, -1)

scaler = StandardScaler()
scaler.fit(X_train_reshaped) # # Обучаю scaler ТОЛЬКО на тренировочных данных

X_train = scaler.transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
X_val = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape)
X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)

print("Форма X_train после добавления канала и нормализации:", X_train.shape)


class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: weight for i, weight in enumerate(class_weights)}


model = build_model(num_classes=8)
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
# Обучение модели
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=64,
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr],
)



# -----------------------------------------------------------------------
# ВИЗУАЛИЗАЦИЯ ДАННЫХ 
# -----------------------------------------------------------------------
y_pred = np.argmax(model.predict(X_test), axis=-1)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Точность модели на тестовых данных: {accuracy:.4f}")
print(f"F1-метрика: {f1:.4f}")
print("Полный отчет классификации:")
print(classification_report(y_test, y_pred, target_names=EMOTIONS.values(), zero_division=0))

# Матрица ошибок (confusion matrix)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTIONS.values(), yticklabels=EMOTIONS.values())
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
plt.ylabel('Точность')
plt.legend()
plt.savefig('loss.png')