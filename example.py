import training
from datetime import datetime
import models
<<<<<<< Updated upstream
import os


=======
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import Callback
import csv
>>>>>>> Stashed changes
SAVE_MODEL = True
MODEL_NAME = 'DNM_densenet201'
PREDICT_IMG = 'NLA_397681429EDR_F0020000AUT_04096M1'
PATH = 'D:\\Projekt Badawczy\\ai4mars-dataset-merged-0.1'

model_map={"URX": models.Unet_resnext50, "LNT": models.Linknet_densenet201, "DFT": models.default, "DNM":models.modelDN201}
accuracy_values = []
loss_values = []
val_accuracy_values = []
val_loss_values = []
model_type = "DFT"
if len(MODEL_NAME)>2 and MODEL_NAME[:3] in model_map:
    model_type=MODEL_NAME[:3]
print(os.path.join(PATH, 'msl\\labels\\train\\'))
print(model_type)
model = model_map[model_type]()
weights_only = True if model else False
model = models.loadModel(MODEL_NAME, model)
<<<<<<< Updated upstream
training.basicTrain(model, epochs=200, n=-1, batch_size=64, learning_rate=0.001, weights_only=weights_only, val_split=0.8, path=PATH)
=======

epochs = 5
history = training.basicTrain(model, epochs, n=40, batch_size=8, learning_rate=0.001, weights_only=weights_only, val_split=0.8)

accuracy_values.extend(history.history['categorical_accuracy'])
loss_values.extend(history.history['loss'])
val_accuracy_values.extend(history.history['val_categorical_accuracy'])
val_loss_values.extend(history.history['val_loss'])
with open('models/plots/accuracy_loss_values.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Epoch', 'Accuracy', 'Loss', 'val_loss', 'val_accuracy'])
    for epoch, accuracy, loss, val_loss, val_accuracy in zip(range(1, epochs + 1), accuracy_values, loss_values, val_loss_values, val_accuracy_values):
            csvwriter.writerow([epoch, accuracy, loss, val_loss, val_accuracy])


>>>>>>> Stashed changes
models.saveModel(model, model_type+"_"+
                        datetime.now().strftime('%d-%m-%y %H-%M-%S') if SAVE_MODEL else 'temp', weights_only=weights_only)
imgs = training.predict(model, PREDICT_IMG, path=PATH)
training.plot(imgs)

