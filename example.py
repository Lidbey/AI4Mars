import training
from datetime import datetime
import models
from stats import calc_stats

SAVE_MODEL = False
CALCULATE_STATS = False  # change to true if you want to plot confusion matrix
MODEL_NAME = 'URX_resnext50'
PREDICT_IMG = 'NLA_397681429EDR_F0020000AUT_04096M1'
model_map={"URX": models.Unet_resnext50, "LNT": models.Linknet_densenet201, "DFT": models.default}

model_type = "DFT"
if len(MODEL_NAME)>2 and MODEL_NAME[:3] in model_map:
    model_type=MODEL_NAME[:3]

print(model_type)
model = model_map[model_type]()
weights_only = True if model else False
model = models.loadModel(MODEL_NAME, model)

training.basicTrain(model, epochs=1, n=16, batch_size=8, learning_rate=0.001, weights_only=weights_only, val_split=0.8)
models.saveModel(model, model_type+"_"+
                        datetime.now().strftime('%d-%m-%y %H-%M-%S') if SAVE_MODEL else 'temp', weights_only=weights_only)

if CALCULATE_STATS:
    prec_rec_stats = calc_stats(model)
else:
    imgs = training.predict(model, PREDICT_IMG)
    training.plot(imgs)
