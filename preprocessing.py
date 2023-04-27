import numpy as np
import tensorflow as tf


def preprocessImage(img):
    #newImg = adjustGamma(img, params)
    #...
    #return newImg
    return img

#def adjustGamma(img, params) itd

def applyMask(img, mask):
    #newImg = zrobcos(img,mask)
    return img


def resize(img, size, type=None):
    if type is None:
        return tf.image.resize(img[..., np.newaxis], size)
    else:
        return tf.image.resize(img[..., np.newaxis], size, type)
