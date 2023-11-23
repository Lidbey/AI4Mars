import os
import time

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
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from training import predict
from pprint import pprint
import time
import pandas as pd


def calc_stats(model, name, date, training_time=None, learning_rate=None, batch_size=None, shape=(128, 128)):
    # count params
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    totalParams = trainableParams + nonTrainableParams

    # calculate model loss on test dataset
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

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=[metrics.mae, metrics.categorical_accuracy])
    eval = model.evaluate(x, to_categorical(y))
    test_loss = eval[0]
    test_mae = eval[1]
    test_acc = eval[2]
    # test_iou = eval[3]

    # calculate average prediction time
    start = time.time()
    predictions = model.predict(x)
    avg_pred_time = (time.time() - start) / len(images)

    # calculate confusion matrix on test dataset
    predictions = np.argmax(predictions, axis=-1)
    true = y.flatten()
    pred = predictions.flatten()

    cm = confusion_matrix(true,
                          pred,
                          labels=[0., 1., 2., 3., 4.])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['soil', 'bedrock', 'sand', 'big rock', 'null'])
    disp.plot()
    if os.path.exists(f"./stats/{name}-{date}/"):
        plt.savefig(f"./stats/{name}-{date}/confusion-matrix.png")
    else:
        os.mkdir(f"./stats/{name}-{date}/")
        plt.savefig(f"./stats/{name}-{date}/confusion-matrix.png")
    plt.close()
    cm_norm = confusion_matrix(true,
                               pred,
                               normalize='true',
                               labels=[0., 1., 2., 3., 4.])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['soil', 'bedrock', 'sand', 'big rock', 'null'])
    disp.plot()
    if os.path.exists(f"./stats/{name}-{date}/"):
        plt.savefig(f"./stats/{name}-{date}/confusion-matrix-normalized.png")
    else:
        os.mkdir(f"./stats/{name}-{date}/")
        plt.savefig(f"./stats/{name}-{date}/confusion-matrix-normalized.png")
    plt.close()

    # prec = {'soil': cm[0, 0] / np.sum(cm[:, 0]),
    #         'bedrock': cm[1, 1] / np.sum(cm[:, 1]),
    #         'sand': cm[2, 2] / np.sum(cm[:, 2]),
    #         'big rock': cm[3, 3] / np.sum(cm[:, 3]),
    #         'null': cm[3, 3] / np.sum(cm[:, 3])
    #         }
    # rec = {'soil': cm[0, 0] / np.sum(cm[0, :]),
    #        'bedrock': cm[1, 1] / np.sum(cm[1, :]),
    #        'sand': cm[2, 2] / np.sum(cm[2, :]),
    #        'big rock': cm[3, 3] / np.sum(cm[3, :]),
    #        'null': cm[4, 4] / np.sum(cm[3, :])
    #        }
    # f_score = {'soil': 2 * prec['soil'] * rec['soil'] / (prec['soil'] + rec['soil']),
    #            'bedrock': 2 * prec['bedrock'] * rec['bedrock'] / (prec['bedrock'] + rec['bedrock']),
    #            'sand': 2 * prec['sand'] * rec['sand'] / (prec['sand'] + rec['sand']),
    #            'big rock': 2 * prec['big rock'] * rec['big rock'] / (prec['big rock'] + rec['big rock']),
    #            'null': 2 * prec['null'] * rec['null'] / (prec['null'] + rec['null'])
    #            }

    if os.path.exists(f"./stats/{name}-{date}/"):
        np.save(f"./stats/{name}-{date}/confusion-matrix", cm)
        np.save(f"./stats/{name}-{date}/confusion-matrix-normalized", cm_norm)
    else:
        os.mkdir(f"./stats/{name}-{date}/")
        np.save(f"./stats/{name}-{date}/confusion-matrix", cm)
        np.save(f"./stats/{name}-{date}/confusion-matrix-normalized", cm_norm)

    stats = {"Parameters": totalParams, "Trainable Parameters": trainableParams,
             "Non Trainable Parameters": nonTrainableParams, "Prediction Time": avg_pred_time,
             "Training Time": training_time, "Learning Rate": learning_rate,
             "Batch Size": batch_size, "Test Loss": test_loss, "Test MAE": test_mae,
             "Test Acc": test_acc}

    # Dodać accuracy dobrze / wszystkie

    df = pd.DataFrame(stats, index=[name])
    df.to_csv(f"./stats/{name}-{date}/stats.csv")
    print(f"Stats saved at ./stats/{name}-{date}/")
