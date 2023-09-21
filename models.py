from tensorflow.keras import layers
import segmentation_models as sm
from keras.layers import Input, Conv2D
from keras.models import Model
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

def Unet_resnext50():
    BACKBONE = 'resnext50'
    preprocess_input = sm.get_preprocessing(BACKBONE)

    base_model = sm.Unet(BACKBONE, classes=5, activation='softmax', encoder_weights='imagenet')


    inp = Input(shape=(128, 128, 1))
    l1 = Conv2D(3, (3, 3), padding='same')(inp) # map N channels data to 3 channels
    out = base_model(l1)

    model = Model(inputs=inp, outputs=out, name=base_model.name)

    model.compile(
        'Adam',
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )
    return model

def Linknet_densenet201():
    BACKBONE = 'densenet201'
    preprocess_input = sm.get_preprocessing(BACKBONE)

    base_model = sm.Linknet(BACKBONE, classes=5, activation='softmax', encoder_weights='imagenet')

    inp = Input(shape=(None, None, 1))
    l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
    out = base_model(l1)

    model = Model(inp, out, name=base_model.name)

    model.compile(
        'Adam',
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )
    return model

def default():
    return None

#def modelv2():

def saveModel(model, name, weights_only=False):
    if weights_only:
        model.save_weights(f'models/{name}')
    else:
        model.save(f'models/{name}')

def loadModel(name, model=None):
    if model is None:
        model = keras.models.load_model(f'models/{name}')
    else:
        model.load_weights(f'models/{name}')
    return model


