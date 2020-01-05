import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import glob
import os
from loguru import logger as log 


def _base_model(alpha=1.0, load_local=True):

    model = tfk.applications.mobilenet_v2.MobileNetV2(
        include_top=False,
        input_shape=(224, 224, 3),
        alpha=alpha,
    )

    if load_local:
        try:
            list_of_files = glob.glob('./weights/base/*.h5')
            latest_file = max(list_of_files, key=os.path.getctime)
            model.load_weights(latest_file)
            log.success(f"Successfully loaded base model from {latest_file}")
        except Exception as ex:
            log.error("Could not load local weights.")
            log.error(ex.message)

    return model


def warmup_model(momentum=0.9, **kwargs):
    base_model = _base_model(**kwargs)
    for layer in base_model.layers:
        if "BatchNormalization" in str(type(layer)):
            layer.trainable = True
            layer.momentum = momentum
        else:
            layer.trainable = False

    model = tfk.Sequential([
        base_model,
        tfkl.GlobalAveragePooling2D(),
        tfkl.Dense(1, activation='sigmoid'),
    ])
    return model


def main_model(dropout=0.7, **kwargs):
    base_model = _base_model(**kwargs)

    model = tfk.Sequential([
        base_model,
        tfkl.GlobalAveragePooling2D(),
        tfkl.Dropout(dropout),
        tfkl.Dense(64, activation='relu'),
        tfkl.Dropout(dropout),
        tfkl.Dense(64, activation='relu'),
        tfkl.Dropout(dropout),
        tfkl.Dense(1, activation='sigmoid'),
    ])
    return model


def freeze_base_model(model, until=7):
    base_model = model.layers[0]

    trainable = False
    for layer in base_model.layers:
        if f"_{start}" in layer.name:
            trainable = True
        layer.trainable = trainable
        if "BatchNormalization" in str(type(layer)):
            layer.trainable = True
    