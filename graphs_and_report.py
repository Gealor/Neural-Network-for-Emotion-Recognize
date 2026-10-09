from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import config


def build_report(y_test, y_pred, groups=None):
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print(f"Точность модели на тестовых данных: {accuracy:.4f}")
    print(f"F1-метрика (weighted): {weighted_f1:.4f}")
    print(f"F1-метрика (macro): {macro_f1:.4f}")
    print("Полный отчет классификации:")
    print(classification_report(y_test, y_pred, target_names=config.EMOTIONS.values(), zero_division=0))

    if groups is not None:
        groups = np.asarray(groups)
        print("Метрики по корпусам:")
        for group in np.unique(groups):
            mask = groups == group
            g_acc = accuracy_score(y_test[mask], y_pred[mask])
            g_wf1 = f1_score(y_test[mask], y_pred[mask], average='weighted', zero_division=0)
            g_mf1 = f1_score(y_test[mask], y_pred[mask], average='macro', zero_division=0)
            print(f"  {str(group):<10} n={int(mask.sum()):<5} "
                  f"acc={g_acc:.4f}  f1_weighted={g_wf1:.4f}  f1_macro={g_mf1:.4f}")

    return {"accuracy": accuracy, "weighted_f1": weighted_f1, "macro_f1": macro_f1}
    

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