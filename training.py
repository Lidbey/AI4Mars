import sys

import keras
import numpy as np
from PIL import ImageOps
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import os
from generator import DataGenerator
import imageio.v3 as iio
from preprocessing import resize


def basicTrain(model, epochs, batch_size=64, n=-1, learning_rate=0.001, save_freq=sys.maxsize):
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    keras.backend.set_value(model.optimizer.learning_rate, learning_rate)
    generator = DataGenerator(n=n, batch_size=batch_size)
    model.fit(generator, epochs=epochs, callbacks=[callbackModelEpoch('models/checkpoints')])

def callbackModelBatch(directory, n=100):
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(directory, 'model_{epoch:02d}_{batch:04d}'),
        save_weights_only=False,
        save_freq=n,
    )
    return checkpoint_callback

def callbackModelEpoch(directory):
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(directory, 'model_{epoch:02d}'),
        save_weights_only=False,
        save_freq='epoch',
    )
    return checkpoint_callback


def predict(model, fileName, shape=(128,128)):
    image_path = 'data/ai4mars-dataset-merged-0.1/msl/images/edr/'
    mask_path = 'data/ai4mars-dataset-merged-0.1/msl/labels/train/'
    xPath = image_path + fileName + '.JPG'
    yPath = mask_path + fileName + '.PNG'
    x = resize(iio.imread(xPath), shape)/255.0
    y = resize(iio.imread(yPath), shape, 'nearest')
    yPred = np.argmax(model.predict(x[np.newaxis, ...]), axis=-1)

    return [x, y, yPred[0][..., np.newaxis]]


def plot(imgs):
    f, axarr = plt.subplots(1, len(imgs))

    # image
    axarr[0].imshow(imgs[0])

    # true labels
    y = imgs[1].numpy()
    y[y == 255] = 4
    y = y * 63.75
    axarr[1].imshow(y)

    # predicted labels
    y = imgs[2]
    y[y == 255] = 4
    y = y * 63.75
    axarr[2].imshow(y)

    plt.show()
