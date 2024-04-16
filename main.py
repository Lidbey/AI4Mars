import training
from datetime import datetime
import models
import os

SAVE_MODEL = True
MODEL_NAME = 'model_16_best_def'
PREDICT_IMG = 'NLB_565366137EDR_F0670604NCAM00257M1'
PATH = 'data/ai4mars-dataset-merged-0.1'

model_map={"URX": models.Unet_resnext50, "LNT": models.Linknet_densenet201, "DFT": models.default, "DNM":models.modelDN201}
model_type = "DFT"
if len(MODEL_NAME)>2 and MODEL_NAME[:3] in model_map:
    model_type=MODEL_NAME[:3]
model = model_map[model_type]()
weights_only = True if model else False
model = models.loadModel(MODEL_NAME, model)
training.basicTrain(model, name = MODEL_NAME, epochs=200, n=-1, batch_size=8, learning_rate=0.001,
                    weights_only=weights_only, val_split=0.8)
models.saveModel(model, model_type+"_"+
                        datetime.now().strftime('%d-%m-%y %H-%M-%S') if SAVE_MODEL else 'temp', weights_only=weights_only)
imgs = training.predict(model, PREDICT_IMG, label_path='data/ai4mars-dataset-merged-0.1/msl/labels/test/masked-gold-min3-100agree/', test=True)

training.plot(imgs)

