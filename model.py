import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from ultralytics import YOLO

IMG_SIZE = (384, 384)
WEIGHTS_PATH = "weights/model.weights.h5"
YOLO_PATH = "weights/last.pt"


def build_model():
    # Reconstrução do modelo EfficientNetV2S
    base_model = EfficientNetV2S(
        include_top=False,
        weights=None,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )

    model_fire = models.Sequential([
        layers.Lambda(preprocess_input),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid", dtype="float32"),
    ])

    model_fire.build((None, 384, 384, 3))
    model_fire.load_weights(WEIGHTS_PATH)

    # Carregamento do modelo YOLO26s treinado
    model_yolo = YOLO(YOLO_PATH)

    return model_fire, model_yolo
