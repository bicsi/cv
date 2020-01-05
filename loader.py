from loguru import logger as log
import numpy as np
import cv2 as cv 
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

def _process_file(file):
    classes = []
    names = []
    with open(file, 'r') as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            n, c = line.split(',')
            names.append(n)
            classes.append(c)
    return np.array(names), np.array(classes)



def load_data(filename, load_images=True, dim='full'):
    names, labels = _process_file(filename)
    labels = np.array([int(x) for x in labels])
    names =  np.array(names)

    if dim == 'tiny':
        names = names[:3000]
        labels = labels[:3000]

    ret = {
        "names": names,
        "labels": labels,
    }
    
    if load_images:
        ret["images"] = _load_images(names)
    
    return ret


def load_train_data(**kwargs):
    return load_data('train_labels.txt', **kwargs)


def build_generators(data, batch_size=32):
    X = data['images'].astype(np.float32)
    y = data['labels'].astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    train_datagen = tfk.preprocessing.image.ImageDataGenerator(
        rescale=(1. / 255.),
        zoom_range=0.1,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='constant',
        cval=0,
    )
    val_datagen = tfk.preprocessing.image.ImageDataGenerator(
        rescale=(1. / 255.),
    )
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

    return train_generator, val_generator


def load_test_data(**kwargs):
    ret = load_data('sample_submission.txt', **kwargs)
    del ret["labels"]
    return ret


def _load_images(names):
    imgs = [cv.imread(f"data/{name}.png") for name in names]
    return np.array(imgs)