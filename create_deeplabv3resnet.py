# All code is from there https://keras.io/examples/vision/deeplabv3_plus/
import keras
from keras.applications.nasnet import layers
import tensorflow as tf
import keras_cv

import models


def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input, input_shape=None):
    dims = dspp_input.shape
    x=None
    if dims[-3] or dims[-2] is None:
        x = layers.AveragePooling2D()(dspp_input)
        x = convolution_block(x, kernel_size=1, use_bias=True)
        out_pool = layers.UpSampling2D(
             interpolation="bilinear",
        )(x)
    else:
        x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
        x = convolution_block(x, kernel_size=1, use_bias=True)
        out_pool = layers.UpSampling2D(
            size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
        )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3PlusResnet50(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)

def DeeplabV3PlusYolov8(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    model = keras_cv.models.YOLOV8Backbone.from_preset(
    "yolo_v8_l_backbone_coco"
    )
    x = model.get_layer("stack4_c2f_output").output
    x = DilatedSpatialPyramidPooling(x, image_size)
    keras.Model(inputs = model_input, outputs=x)
    input_a = None
    if x.shape[1] is None or x.shape[2] is None:
        input_a = layers.UpSampling2D(
            interpolation="bilinear",
        )(x)
    else:
        input_a = layers.UpSampling2D(
            size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
            interpolation="bilinear",
        )(x)
    input_b = model.get_layer("stack2_c2f_output").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    y = None
    if x.shape[1] is None or x.shape[2] is None:
        y = layers.UpSampling2D(
            interpolation="bilinear",
        )(x)
    else:
        y = layers.UpSampling2D(
            size=(image_size // x.shape[1], image_size // x.shape[2]),
            interpolation="bilinear",
        )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(y)
    return keras.Model(inputs=model_input, outputs=model_output)


model = DeeplabV3PlusResnet50(image_size=128, num_classes=5)
models.saveModel(model, 'model_resnet50')
#model.summary()

#model2 = DeeplabV3PlusYolov8(image_size=128, num_classes=5)