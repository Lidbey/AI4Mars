from tensorflow.keras import layers
import keras

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

#def modelv2():

def saveModel(model, name):
    model.save(f'models/{name}')

def loadModel(name):
    model = keras.models.load_model(f'models/{name}')
    return model