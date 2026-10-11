import csv
import json
import sys
from datetime import datetime

import keras
import numpy as np
from keras import layers, losses, models, optimizers, regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import compute_class_weight

import config
from graphs_and_report import (
    build_accuracy_graph,
    build_confusion_matrix,
    build_loss_graph,
    build_report,
)
from logging_utils import Tee

SEED = 42
keras.utils.set_random_seed(SEED)

SSL_MODEL = "wavlm-base-plus"


LOG_DIR = config.RESULTS_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_TS = f"{datetime.now():%Y%m%d_%H%M%S}"
log_path = LOG_DIR / f"train_ssl_{SSL_MODEL}_{RUN_TS}.log"
log_file = open(log_path, "w", encoding="utf-8")

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

print(f"Логирование вывода в файл: {log_path}")


def build_head(input_dim, num_classes):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.BatchNormalization()(inputs)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0003),
        loss=losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    return model


def load_split(model_key, type_dataset):
    emb_path = config.SSL_OUTPUT_DIR / "embeddings" / f"{model_key}_{type_dataset}.npy"
    y_path = config.SSL_OUTPUT_DIR / f"y_{type_dataset}.npy"
    X = np.load(emb_path)
    y = np.load(y_path).astype(int)
    return X, y


batch_size = 32
X_train, y_train = load_split(SSL_MODEL, "train")
X_val, y_val = load_split(SSL_MODEL, "val")
X_test, y_test = load_split(SSL_MODEL, "test")

print(f"Модель-энкодер: {SSL_MODEL}")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: weight for i, weight in enumerate(class_weights)}

num_classes = len(config.EMOTIONS.keys())
y_train_oh = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_val_oh = keras.utils.to_categorical(y_val, num_classes=num_classes)

model = build_head(input_dim=X_train.shape[1], num_classes=num_classes)
model.summary()

output_dir = config.RESULTS_DIR / "ssl"
output_dir.mkdir(parents=True, exist_ok=True)

early_stop = EarlyStopping(monitor='val_accuracy', mode='max', patience=8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', mode='max', factor=0.2, patience=6, min_lr=0.00001)
ckpt = ModelCheckpoint(str(output_dir / f"best_head_{SSL_MODEL}.keras"), monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

# Обучение модели
history = model.fit(
    X_train, y_train_oh,
    batch_size=batch_size,
    epochs=100,
    validation_data=(X_val, y_val_oh),
    class_weight=class_weights,
    callbacks=[
        early_stop,
        reduce_lr,
        ckpt
    ],
)

print("\nОценка на тестовых данных...")
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=-1)

print(f"Размер предсказаний (y_pred): {y_pred.shape}")
print(f"Размер истинных меток (y_test): {y_test.shape}")

groups_test_path = config.SSL_OUTPUT_DIR / 'groups_test.npy'
groups_test = np.load(groups_test_path) if groups_test_path.exists() else None

metrics = build_report(y_test, y_pred, groups=groups_test)

# Матрица ошибок (confusion matrix)
build_confusion_matrix(y_test, y_pred, output_dir=output_dir)

# График точности на тренировочных и валидационных данных
build_accuracy_graph(history, output_dir=output_dir)

# График потерь
build_loss_graph(history, output_dir=output_dir)

# Дамп истории обучения + строка в общий журнал прогонов (для сравнения фаз)
with open(LOG_DIR / f"history_ssl_{SSL_MODEL}_{RUN_TS}.json", "w", encoding="utf-8") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

runs_csv = config.RESULTS_DIR / "runs.csv"
write_header = not runs_csv.exists()
with open(runs_csv, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["timestamp", "accuracy", "weighted_f1", "macro_f1", "epochs", "notes"])
    writer.writerow([
        RUN_TS,
        f"{metrics['accuracy']:.4f}",
        f"{metrics['weighted_f1']:.4f}",
        f"{metrics['macro_f1']:.4f}",
        len(history.history.get("loss", [])),
        f"ssl_head:{SSL_MODEL}",
    ])
