from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

import config


def build_report(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')   
    print(f"Точность модели на тестовых данных: {accuracy:.4f}")
    print(f"F1-метрика: {f1:.4f}")
    print("Полный отчет классификации:")
    print(classification_report(y_test, y_pred, target_names=config.EMOTIONS.values(), zero_division=0))
    

def build_confusion_matrix(y_test, y_pred, output_dir: Path = config.RESULTS_DIR):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=config.EMOTIONS.values(), yticklabels=config.EMOTIONS.values())
    plt.ylabel('Истинные значения')
    plt.xlabel('Предсказанные значения')
    plt.title('Confusion Matrix')
    plt.savefig(output_dir / 'confusion_matrix.png')


def build_accuracy_graph(history, output_dir: Path = config.RESULTS_DIR):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Тренировочная точность')
    plt.plot(history.history['val_accuracy'], label='Валидационная точность')
    plt.title('Точность модели на тренировочных и валидационных данных')
    plt.xlabel('Эпохи')
    plt.ylabel('Точность')
    plt.legend()
    plt.savefig(output_dir / 'accuracy.png')


def build_loss_graph(history, output_dir: Path = config.RESULTS_DIR):
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label='Тренировочные потери')
    plt.plot(history.history['val_loss'], label='Валидационные потери')
    plt.title('Потери модели на тренировочных и валидационных данных')
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.legend()
    plt.savefig(output_dir / 'loss.png')