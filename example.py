import glob
import os
import imageio.v3 as iio
import keras.backend
from PIL import ImageOps
from PIL.Image import Image
from PIL.ImageShow import show
from PIL._imaging import display
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

import models
import training


def loadDataWIP(img_size, amount, channels=1):
    dir = 'data/ai4mars-dataset-merged-0.1/msl/'
    labelsTrainDir = dir + 'labels/train/'
    photosDir = dir + 'images/edr/'
    inputData = []
    outputData = []
    i = 0
    for labelPath in glob.iglob(f'{labelsTrainDir}/*'):
        label = iio.imread(labelPath).copy()
        label[label == 255] = 4
        labelName = os.path.basename(labelPath)
        photoName = os.path.splitext(labelName)[0] + '.JPG'
        photoPath = photosDir + photoName
        photo = iio.imread(photoPath) / 255.0

        photo = np.expand_dims(photo, -1)
        photo = photo.repeat(channels, axis=-1)  # additional channels are copies of the original grayscale channel

        inputData.append(photo)
        outputData.append(label)
        if i == amount:
            break
        i = i + 1

    inputArr = tf.image.resize(np.array(inputData), img_size)
    outputArr = tf.image.resize(np.array(outputData)[..., np.newaxis], img_size, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    outputArr = to_categorical(outputArr)

    return inputArr, outputArr


# model = models.modelv1((128, 128), 5)
model = models.segmentation_models_unet((128, 128, 3), 5)
input, output = loadDataWIP((128, 128), 3)
training.basicTrain(model, input, output, 30, 3)

predicted_id = 2
photo = input[predicted_id]
label = np.argmax(output[predicted_id], axis=-1)[...,np.newaxis]
predicted = np.argmax(model.predict(photo[np.newaxis,...])[0], axis=-1)[...,np.newaxis]

imgs = [photo, label, predicted]
f, axarr = plt.subplots(1,3)
for i, img in enumerate(imgs):
    axarr[i].imshow(ImageOps.autocontrast(keras.utils.array_to_img(img)))

axarr[0].title.set_text('Real')
axarr[1].title.set_text('Labels')
axarr[2].title.set_text('Predicted Labels')
plt.show()
