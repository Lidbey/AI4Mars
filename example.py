import glob
import os
import imageio.v3 as iio
import keras
import keras.backend
from PIL import ImageOps
from PIL.Image import Image
from PIL.ImageShow import show
from PIL._imaging import display
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

img_size = (128,128)
num_classes = 5
MAX=1
PREDICT = 1

def createNetworkManual():
    inputs = keras.Input(shape = img_size+(1,))
    x = layers.Conv2D(32, 3, strides = 2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
#https://keras.io/examples/vision/oxford_pets_image_segmentation/
    prev = x

    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            prev
        )
        x = layers.add([x, residual])  # Add back residual
        prev = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

            # Project residual
        residual = layers.UpSampling2D(2)(prev)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        prev = x  # Set aside next residual

        # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

        # Define the model
    model = keras.Model(inputs, outputs)
    return model

dir = 'data/ai4mars-dataset-merged-0.1/msl/'
labelsTrainDir = dir+'labels/train/'
photosDir = dir+'images/edr/'


f, axarr = plt.subplots(1,3)

model = createNetworkManual()

inputData = []
outputData = []
i = 0
for labelPath in glob.iglob(f'{labelsTrainDir}/*'):
    label = iio.imread(labelPath).copy()
    label[label==255] = 4
    labelName = os.path.basename(labelPath)
    photoName = os.path.splitext(labelName)[0]+'.JPG'
    photoPath = photosDir + photoName
    photo = iio.imread(photoPath)/255.0
    inputData.append(photo)
    outputData.append(label)
    if i == PREDICT:
        print(labelPath)
        axarr[0].imshow((photo*255)[...,np.newaxis])
        axarr[1].imshow(ImageOps.autocontrast(keras.utils.array_to_img(label[...,np.newaxis])))
    if i == MAX:
        break
    i = i + 1

inputArr = tf.image.resize(np.array(inputData)[...,np.newaxis], img_size)
outputArr = tf.image.resize(np.array(outputData)[...,np.newaxis], img_size, tf.image.ResizeMethod.NEAREST_NEIGHBOR)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
keras.backend.set_value(model.optimizer.learning_rate, 0.00001)
model.fit(x=inputArr, y=outputArr, batch_size=1, epochs=400)

# Display mask predicted by our model
preds = model.predict(inputArr[PREDICT][np.newaxis,...])
mask = np.argmax(preds, axis=-1)
mask = np.expand_dims(mask, axis=-1)
img = ImageOps.autocontrast(keras.utils.array_to_img(mask[0]))
axarr[2].imshow(img)

plt.show()

