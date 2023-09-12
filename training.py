import sys
import glob
import keras
import numpy as np
from PIL import ImageOps
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import os
from tensorflow import metrics
from generator import DataGenerator
import imageio as iio
from preprocessing import resize

def basicTrain(model, epochs, batch_size=64, n=-1, learning_rate=0.001,  save_freq=sys.maxsize, weights_only=False, val_split=0.8):
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=[metrics.mae, metrics.categorical_accuracy])
    keras.backend.set_value(model.optimizer.learning_rate, learning_rate)
    x = -1
    if n ==-1:
        cnt=0
        for labelPath in glob.iglob('data/ai4mars-dataset-merged-0.1/msl/labels/train/'):
            cnt = cnt+1
        x=cnt*val_split
    else:
        x=n*val_split
    generator = DataGenerator(n1=0, n2=x, batch_size=batch_size)
    generator_val = DataGenerator(n1=x, n2=-1, batch_size=batch_size)
    model.fit(generator, epochs=epochs, callbacks=[callbackModelEpoch('models/checkpoints', weights_only)], validation_data = generator_val)


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
    mask_path = 'data/ai4mars-dataset-merged-0.1/msl/labels/train/'
    xPath = image_path + fileName + '.JPG'
    yPath = mask_path + fileName + '.PNG'
    x = resize(iio.imread(xPath), shape)/255.0
    y = resize(iio.imread(yPath), shape)
    yPred = np.argmax(model.predict(x[np.newaxis, ...]), axis=-1)

    return [x, y, yPred[0][..., np.newaxis]]


def plot(imgs):
    f, axarr = plt.subplots(1, len(imgs))
    for i, img in enumerate(imgs):
        axarr[i].imshow(ImageOps.autocontrast(keras.utils.array_to_img(img)))
    plt.show()

