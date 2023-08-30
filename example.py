import models
import training
from datetime import datetime
import visualkeras


SAVE_MODEL = False
MODEL_NAME = 'model_resnet50'
PREDICT_IMG = 'NLA_397681339EDR_F0020000AUT_04096M1'

model = models.loadModel(MODEL_NAME)
training.basicTrain(model, epochs=50, n=4, batch_size=4, learning_rate=0.001)
models.saveModel(model, datetime.now().strftime('%d-%m-%y %H-%M-%S') if SAVE_MODEL else 'temp')
imgs = training.predict(model, PREDICT_IMG)
training.plot(imgs)

