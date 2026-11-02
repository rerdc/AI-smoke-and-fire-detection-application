import matplotlib.pyplot as plt
import numpy as np
import keras

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model import build_model

# Configurações iniciais
IMG_SIZE = (384, 384)
BATCH_SIZE = 8

VAL_DIR = "data/FlameVision/val"
TEST_DIR = "data/FlameVision/test"

# Carregamento dos pesos do modelo final
two_models = build_model()
model = two_models[0]

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
    ],
)

# Carregamento dos conjuntos de dados
val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    label_mode="binary",
    class_names=["No_Fire", "Fire"],
    shuffle=False,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    label_mode="binary",
    class_names=["No_Fire", "Fire"],
    shuffle=False,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

# =========================================================
# Cálculo do limiar a partir do conjunto de validação
# =========================================================

print("\nA determinar o melhor limiar no conjunto de validação...\n")

y_val_true = []
y_val_pred = []

# Recolha das previsões do conjunto de validação
for x, y in val_ds:
    preds = model.predict(x, verbose=0)
    y_val_true.extend(y.numpy())
    y_val_pred.extend(preds)

y_val_true = np.array(y_val_true).flatten()
y_val_pred = np.array(y_val_pred).flatten()

# Determinação do limiar em função da melhor pontuação F1
thresholds = np.arange(0.1, 0.9, 0.01)
best_thresh = 0.5
best_f1 = 0

for t in thresholds:
    y_hat = (y_val_pred >= t).astype(int)

    tp = np.sum((y_hat == 1) & (y_val_true == 1))
    fp = np.sum((y_hat == 1) & (y_val_true == 0))
    fn = np.sum((y_hat == 0) & (y_val_true == 1))

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print("Melhor limiar:", best_thresh)
print("Pontuação F1 correspondente:", best_f1)

# =========================================================
# Avaliação final no conjunto de teste
# =========================================================

print("\nAvaliação no conjunto de teste...\n")

results = model.evaluate(test_ds, verbose=1)

for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

# Recolha das previsões do conjunto de teste
y_test_true = []
y_test_pred = []

for x, y in test_ds:
    preds = model.predict(x, verbose=0)
    y_test_true.extend(y.numpy())
    y_test_pred.extend(preds)

y_test_true = np.array(y_test_true).flatten()
y_test_pred = np.array(y_test_pred).flatten()

# Cálculo de F1 no teste com o limiar selecionado
y_test_labels = (y_test_pred >= best_thresh).astype(int)

tp = np.sum((y_test_labels == 1) & (y_test_true == 1))
fp = np.sum((y_test_labels == 1) & (y_test_true == 0))
fn = np.sum((y_test_labels == 0) & (y_test_true == 1))

test_precision = tp / (tp + fp + 1e-7)
test_recall = tp / (tp + fn + 1e-7)
test_f1 = 2 * test_precision * test_recall / (
    test_precision + test_recall + 1e-7
)

print("\nPontuação F1 no conjunto de teste:", test_f1)

# Matriz de confusão
print("\nA gerar a matriz de confusão...\n")

cm = confusion_matrix(
    y_test_true,
    y_test_labels,
)

print("Matriz de confusão:\n", cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No_Fire", "Fire"],
)

disp.plot(cmap=plt.cm.Blues)
plt.title(f"Matriz de confusão (limiar={best_thresh:.2f})")
plt.show()
