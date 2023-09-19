import sys
import keras
import numpy as np
from PIL import ImageOps
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import os
from tensorflow import metrics
from generator_manager import DataManager
import imageio as iio
from preprocessing import resize
import tensorflow as tf

def basicTrain(model, epochs, batch_size=64, n=-1, learning_rate=0.001,  save_freq=sys.maxsize, weights_only=False, val_split=0.8):
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=[metrics.mae, metrics.categorical_accuracy])
    keras.backend.set_value(model.optimizer.learning_rate, learning_rate)
    generator = DataManager(val_split=val_split, batch_size=batch_size, n=n)
    training_generator, val_generator = generator.get()
    model.fit(training_generator, epochs=epochs, callbacks=[callbackModelEpoch('models/checkpoints', weights_only)], validation_data = val_generator)


def callbackModelBatch(directory, n=100, weights_only=False):
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(directory, 'model_{epoch:02d}_{batch:04d}'),
        save_weights_only=weights_only,
        save_freq=n,
    )
    return checkpoint_callback

def callbackModelEpoch(directory, weights_only=False):
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(directory, 'model_{epoch:02d}'),
        save_weights_only=weights_only,
        save_freq='epoch',
    )
    return checkpoint_callback


def predict(model, fileName, shape=(128,128)):
    image_path = 'data/ai4mars-dataset-merged-0.1/msl/images/edr/'
    label_path = 'data/ai4mars-dataset-merged-0.1/msl/labels/train/'
    yPath = label_path + fileName + '.PNG'

    label = iio.imread(yPath)
    y = resize(label, shape, tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    photoPath = image_path + fileName + '.JPG'
    photo = iio.imread(photoPath)
    x = resize(photo, shape)
    #y = to_categorical(y)
    x = x / 255.0
    y = y.__array__()
    y[y == 255] = 4
    yPred = np.argmax(model.predict(np.array([x])), axis=-1)

    return [x, y, yPred[0][..., np.newaxis]]


def plot(imgs):
    f, axarr = plt.subplots(1, len(imgs))
    for i, img in enumerate(imgs):
        axarr[i].imshow(ImageOps.autocontrast(keras.utils.array_to_img(img)))
    plt.show()

