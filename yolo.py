import glob

import models
import os


model = models.modelYOLOv8seg()
i = 0
dir = 'data/ai4mars-dataset-merged-0.1/msl/'
labelsTrainDir = dir + 'labels/train/'
photosDir = dir + 'images/edr/'
inputData = []
outputData = []
i = 0
amount = 10
for labelPath in glob.iglob(f'{labelsTrainDir}/*'):
    labelName = os.path.basename(labelPath)
    photoName = os.path.splitext(labelName)[0] + '.JPG'
    photoPath = photosDir + photoName
    inputData.append(photoName)
    outputData.append(labelName)
    if i == amount:
        break
    i = i + 1
model.train(data='training.yaml')
x = model.predict('data/ai4mars-dataset-merged-0.1/msl/images/edr/NLA_397681339EDR_F0020000AUT_04096M1.JPG')

print('x')