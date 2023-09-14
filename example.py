import training
from datetime import datetime
import models


SAVE_MODEL = True
MODEL_NAME = 'URX_model_118'
PREDICT_IMG = 'NLA_397681455EDR_F0020000AUT_04096M1'
model_map={"URX": models.Unet_resnext50(), "DFT": None}

model_type = "DFT"
if len(MODEL_NAME)>2 and MODEL_NAME[:3] in model_map:
    model_type=MODEL_NAME[:3]

print(model_type)
model = model_map[model_type]
model = models.loadModel(MODEL_NAME, model)
weights_only = True if model else False

training.basicTrain(model, epochs=300, n=32, batch_size=32, learning_rate=0.0001, weights_only=weights_only, val_split=1)
models.saveModel(model, model_type+"_"+
                        datetime.now().strftime('%d-%m-%y %H-%M-%S') if SAVE_MODEL else 'temp', weights_only=weights_only)
imgs = training.predict(model, PREDICT_IMG)
training.plot(imgs)
