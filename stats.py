import os

import numpy as np
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
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from training import predict
from pprint import pprint


def calc_stats(model, shape=(128, 128)):
    image_path = 'data/ai4mars-dataset-merged-0.1/msl/images/edr/'
    label_path = 'data/ai4mars-dataset-merged-0.1/msl/labels/test/masked-gold-min1-100agree/'

    labels = []
    images = []
    for filename in os.listdir(label_path):
        yPath = label_path + filename
        xPath = image_path + filename[:-11] + '.JPG'
        labels.append(iio.imread(yPath))
        images.append(iio.imread(xPath))
    labels = np.array(labels)
    images = np.array(images)

    x = np.zeros(shape=(len(labels), shape[0], shape[1], 1))
    y = np.zeros(shape=(len(labels), shape[0], shape[1], 1))
    for i in range(len(labels)):
        y[i] = resize(labels[i], shape, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x[i] = resize(images[i], shape)

    y[y == 255] = 4
    x = x / 255.0

    predictions = model.predict(x)
    predictions = np.argmax(predictions, axis=-1)
    true = y.flatten()
    pred = predictions.flatten()

    cm = confusion_matrix(true,
                          pred,
                          labels=[0., 1., 2., 3., 4.])
    cm_norm = confusion_matrix(true,
                               pred,
                               normalize='true',
                               labels=[0., 1., 2., 3., 4.])

    prec = {'soil': cm[0, 0] / np.sum(cm[:, 0]),
            'bedrock': cm[1, 1] / np.sum(cm[:, 1]),
            'sand': cm[2, 2] / np.sum(cm[:, 2]),
            'big rock': cm[3, 3] / np.sum(cm[:, 3]),
            'null': cm[3, 3] / np.sum(cm[:, 3])
            }
    rec = {'soil': cm[0, 0] / np.sum(cm[0, :]),
           'bedrock': cm[1, 1] / np.sum(cm[1, :]),
           'sand': cm[2, 2] / np.sum(cm[2, :]),
           'big rock': cm[3, 3] / np.sum(cm[3, :]),
           'null': cm[4, 4] / np.sum(cm[3, :])
           }
    f_score = {'soil': 2 * prec['soil'] * rec['soil'] / (prec['soil'] + rec['soil']),
               'bedrock': 2 * prec['bedrock'] * rec['bedrock'] / (prec['bedrock'] + rec['bedrock']),
               'sand': 2 * prec['sand'] * rec['sand'] / (prec['sand'] + rec['sand']),
               'big rock': 2 * prec['big rock'] * rec['big rock'] / (prec['big rock'] + rec['big rock']),
               'null': 2 * prec['null'] * rec['null'] / (prec['null'] + rec['null'])
               }

    pprint(f_score)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['soil', 'bedrock', 'sand', 'big rock', 'null'])
    disp.plot(cmap='Blues')

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm,
                                  display_labels=['soil', 'bedrock', 'sand', 'big rock', 'null'])
    disp.plot(cmap='Blues')
    plt.show()

    return cm, cm_norm, prec, rec, f_score
