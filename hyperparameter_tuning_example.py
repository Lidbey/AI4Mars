import keras_tuner
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import os
import imageio.v3 as iio
import keras.backend

from PIL import ImageOps
from keras.utils.np_utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers

def loadDataWIP(img_size, amount):
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
        inputData.append(photo)
        outputData.append(label)
        if i == amount:
            break
        i = i + 1

    inputArr = tf.image.resize(np.array(inputData)[..., np.newaxis], img_size)
    outputArr = tf.image.resize(np.array(outputData)[..., np.newaxis], img_size, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    outputArr = to_categorical(outputArr)

    return inputArr, outputArr


def modelv1(hp):
    img_size = (128, 128)
    num_classes = 5

    inputs = keras.Input(shape = img_size+(1,))
    x = layers.Conv2D(32, 3, strides = 2, padding="same")(inputs)
    x = layers.Activation("relu")(x)
    prev = x

    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            prev
        )
        x = layers.add([x, residual])
        prev = x

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)

        x = layers.UpSampling2D(2)(x)

        residual = layers.UpSampling2D(2)(prev)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])
        prev = x

    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    model = keras.Model(inputs, outputs)

    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


input, output = loadDataWIP((128, 128), 100)

tuner = keras_tuner.RandomSearch(
    hypermodel=modelv1,
    objective="accuracy",
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)

tuner.search(input, output, epochs=3)
print("\n\nTuner search done")
best_hps = tuner.get_best_hyperparameters(5)
model = modelv1(best_hps[0])
model.fit(input, output, epochs=10)

predicted_id = 2
photo = input[predicted_id]
label = np.argmax(output[predicted_id], axis=-1)[...,np.newaxis]
predicted = np.argmax(model.predict(photo[np.newaxis,...])[0], axis=-1)[...,np.newaxis]

imgs = [photo, label, predicted]
f, axarr = plt.subplots(1, 3)
for i, img in enumerate(imgs):
    axarr[i].imshow(ImageOps.autocontrast(keras.utils.array_to_img(img)))

axarr[0].title.set_text('Real')
axarr[1].title.set_text('Labels')
axarr[2].title.set_text('Predicted Labels')
plt.show()
