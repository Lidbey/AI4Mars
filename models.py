from tensorflow.keras import layers
import tensorflow as tf
import keras
from keras_cv_attention_models import yolov8
from ultralytics import YOLO

MODEL_PATH = 'models/'

def modelv1(img_size, num_classes):
    inputs = keras.Input(shape = img_size+(1,))
    x = layers.Conv2D(32, 3, strides = 2, padding="same")(inputs)
    #x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
#https://keras.io/examples/vision/oxford_pets_image_segmentation/
    prev = x

    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        #x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        #x = layers.BatchNormalization()(x)

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
        #x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        #x = layers.BatchNormalization()(x)

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

def modelYOLOv8(inputShape):
    model = keras.Sequential()
    model.add(yolov8.YOLOV8_N(inputShape))
    #model.add(layers.Dense(128, activation='softmax'))
    return model

def modelYOLOv8seg():
    model = YOLO('yolov8n-seg.yaml').load('/models/yolov8n-seg.pt')
    return model

#def modelv2():
"""
import os
from tensorflow.keras.models import model_from_json

def saveModel(directory):
    filename = 'modelv1_'
    i = 1
    while os.path.exists(os.path.join(directory, f"{filename}{i}.py")):
        i += 1
    name = f"{filename}{i}"
    filepath = os.path.join(directory, f"{name}.py")
    with open(filepath, 'w') as f:
        f.write(inspect.getsource(modelv1))
    
    architecture_path = os.path.join(directory, f"{name}.json")
    with open(architecture_path, 'w') as f:
        f.write(model.to_json())
    
    weights_path = os.path.join(directory, f"{name}.h5")
    model.save_weights(weights_path)
"""

"""
from tensorflow.keras.models import load_model

def loadModel(directory)
    with open(f"{directory}.json", 'r') as f:
        model_architecture = f.read()
    model = model_from_json(model_architecture)
    
    model.load_weights(f"{directory}.h5")
    
    return model
"""

"""
from tensorflow.keras.callbacks import ModelCheckpoint
import os

def callbackModel(model, directory):
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(directory, 'model_{epoch:02d}_{batch:04d}.h5'),
        save_weights_only=True,
        save_freq=100,
    )
    return checkpoint_callback
"""
