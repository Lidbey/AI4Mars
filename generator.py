import glob
import numpy as np
import tensorflow as tf
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical
import os
import imageio.v3 as iio

class DataGenerator(Sequence):
    def __init__(self,
                 image_path = 'data/ai4mars-dataset-merged-0.1/msl/images/edr/',
                 mask_path = 'data/ai4mars-dataset-merged-0.1/msl/labels/train/',
                 batch_size=32,
                 dim=(128, 128), n_channels=1,
                 n_classes=5, shuffle=True, n=-1):

        self.list_IDs = []
        self.dim = dim
        self.batch_size = batch_size
        self.image_path = image_path
        self.mask_path = mask_path
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.n = n

        for labelPath in glob.iglob(f'{mask_path}/*'):
            labelName = os.path.basename(labelPath)
            photoName = os.path.splitext(labelName)[0]
            self.list_IDs.append(photoName)

        if n != -1:
            self.list_IDs = self.list_IDs[0:n]
        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):

        X = []
        y = []

        for i, ID in enumerate(list_IDs_temp):

            labelpath = self.mask_path + ID + '.PNG'
            label = iio.imread(labelpath).copy()
            label[label == 255] = 4

            photoPath = self.image_path + ID + '.JPG'
            photo = iio.imread(photoPath) / 255.0

            photo = [photo, photo, photo]

            X.append(photo)
            y.append(label)

        Xarr = tf.image.resize(np.transpose(np.array(X), (0,3,2,1)), self.dim)
        yarr = tf.image.resize(np.array(y)[..., np.newaxis], self.dim,
                               tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        #yarr = to_categorical(yarr)

        return Xarr, yarr
