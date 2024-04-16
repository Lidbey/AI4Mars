import numpy as np
import tensorflow as tf
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical
import os
import imageio as iio

from project.preprocessing import resize


def map_searchsort(arr):
    from_values = np.unique(arr)
    to_values = np.array([0.8823595939650299,
                          0.5916676724364482,
                          2.0119759159923265,
                          4.581654777828741,
                          0.1178505158231011])
    sort_idx = np.argsort(from_values)
    idx = np.searchsorted(from_values, arr, sorter=sort_idx)
    out = to_values[sort_idx][idx]
    return out


class DataGenerator(Sequence):
    def __init__(self,
                 list_IDs,
                 path = 'data\\ai4mars-dataset-merged-0.1',
                 batch_size=32,
                 dim=(128, 128), n_channels=1,
                 n_classes=5, shuffle=True):
        image_path = os.path.join(path,'msl\\images\\edr\\')
        label_path = os.path.join(path,'msl\\labels\\train\\')
        if len(list_IDs) == 0:
            raise Exception("0 items in set, check sets file")
        self.list_IDs = list_IDs
        self.dim = dim
        self.batch_size = batch_size
        self.image_path = image_path
        self.label_path = label_path
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle

        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        x, y, w = self.__data_generation(list_IDs_temp)
        return x, y, w

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):

        x = np.zeros(shape=(len(list_IDs_temp), self.dim[0], self.dim[1], 1))
        y = np.zeros(shape=(len(list_IDs_temp), self.dim[0], self.dim[1], 1))
        w = np.zeros(shape=(len(list_IDs_temp), self.dim[0], self.dim[1]))
        for i, ID in enumerate(list_IDs_temp):

            labelPath = self.label_path + ID + '.PNG'
            label = iio.imread(labelPath)
            y[i] = resize(label, self.dim, tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            photoPath = self.image_path + ID + '.JPG'
            photo = iio.imread(photoPath)
            x[i] = resize(photo, self.dim)

            w[i] = map_searchsort(y[i, :, :, 0])

        y[y == 255] = 4
        y = to_categorical(y)
        x = x / 255.0

        return x, y, w
