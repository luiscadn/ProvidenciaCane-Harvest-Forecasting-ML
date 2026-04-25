from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns


def calcular_metricas(y_true, y_pred):
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)

    print("--- REPORTE DE RESULTADOS ---")
    print(f"F1-Score (Macro) : {macro_f1:.4f}")
    print(f"Cohen's Kappa    : {kappa:.4f}")

    return macro_f1, kappa


def graficar_confusion(y_true, y_pred, labels_target):
    cm = confusion_matrix(y_true, y_pred, labels=labels_target)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels_target,
        yticklabels=labels_target
    )

    plt.title("Matriz de Confusión de la Predicción")
    plt.ylabel("Valor Real (Campo)")
    plt.xlabel("Predicción del Modelo")
    plt.show()

    return cm


def evaluar_modelo(y_true, y_pred, labels_target=['Bajo', 'Medio', 'Alto']):
    macro_f1, kappa = calcular_metricas(y_true, y_pred)
    cm = graficar_confusion(y_true, y_pred, labels_target)

    return macro_f1, kappa, cm