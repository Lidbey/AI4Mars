import models
import training
from datetime import datetime


SAVE_MODEL = False
MODEL_NAME = 'mobilenet_v2_unet'
PREDICT_IMG = 'NLA_397681339EDR_F0020000AUT_04096M1'

model = models.loadModel(MODEL_NAME)
training.basicTrain(model, epochs=1, n=-1, batch_size=32, learning_rate=0.001, n_channels=3)
models.saveModel(model, datetime.now().strftime('%d-%m-%y %H-%M-%S') if SAVE_MODEL else 'temp')
imgs = training.predict(model, PREDICT_IMG)
training.plot(imgs)
