import numpy as np
import tensorflow as tf


def preprocessImage(img):
    #newImg = adjustGamma(img, params)
    #...
    #return newImg
    return img

#def adjustGamma(img, params) itd

def applyMask(image, mask):

    size = image.shape          #Dimension of image

    for i in range(size[0]):
        for j in range(size[1]):
            pi = image[i, j]    #Pixel in image
            pm = mask[i, j]     #Pixel in mask
            for k in range(3):
                if pm[k] != 0:  #Changing image pixel colour only if mask pixel is black
                    pi[k] = 0

            image[i, j] = pi

    return image


def resize(img, size, type=None):
    if type is None:
        return tf.image.resize(img[..., np.newaxis], size)
    else:
        return tf.image.resize(img[..., np.newaxis], size, type)
