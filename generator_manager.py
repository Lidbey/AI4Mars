import glob
import os
import random
import numpy as np
from generator import DataGenerator
import pickle



class DataManager():
    def __init__(self, val_split=0.8, batch_size=32, n=-1, data_path='data/ai4mars-dataset-merged-0.1/msl/labels/train/'):

        self.name = str(n) + '_' + str(val_split)
        self.batch_size=batch_size

        if os.path.exists(f'sets/{self.name}.pickle'):
            self.load()
        else:
            photo_names = []
            cnt = 0
            for labelPath in glob.iglob(f'{data_path}/*'):
                cnt = cnt + 1
                labelName = os.path.basename(labelPath)
                photoName = os.path.splitext(labelName)[0]
                photo_names.append(photoName)
                if cnt >= n and n != -1:
                    break
            training_names = random.sample(photo_names, round(val_split*len(photo_names)))
            val_names = np.setdiff1d(photo_names, training_names)\

            self.training_generator = DataGenerator(list_IDs=training_names, batch_size=batch_size)
            self.val_generator = DataGenerator(list_IDs=val_names, batch_size=batch_size)
            self.save()

    def save(self):
        if not os.path.exists('sets'):
            os.makedirs('sets')
        with open (f'sets/{self.name}.pickle', 'wb+') as handle:
            pickle.dump((self.training_generator.list_IDs, self.val_generator.list_IDs), handle)


    def load(self):
        with open (f'sets/{self.name}.pickle', 'rb') as handle:
            generators = pickle.load(handle)
            self.training_generator = DataGenerator(list_IDs=generators[0], batch_size=self.batch_size)
            self.val_generator = DataGenerator(list_IDs=generators[1], batch_size=self.batch_size)

    def get(self):
        return (self.training_generator, self.val_generator)


