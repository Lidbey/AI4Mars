import numpy as np
import tensorflow as tf

def applyMask(image, mask):

    size = image.shape

    for i in range(size[0]):
        for j in range(size[1]):
            pi = image[i, j]
            pm = mask[i, j]
            for k in range(3):
                if pm[k] != 0:
                    pi[k] = 0

            image[i, j] = pi

    return image

def resize(img, size, type=None):
    if type is None:
        return tf.image.resize(img[..., np.newaxis], size)
    else:
        return tf.image.resize(img[..., np.newaxis], size, type)
