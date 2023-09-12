import training
from datetime import datetime
import models


SAVE_MODEL = True
MODEL_NAME = 'URX_resnext50'
PREDICT_IMG = 'NLA_404684725EDR_F0050104NCAM00107M1'
model_map={"URX": models.Unet_resnext50(), "DFT": None}

model_type = "DFT"
if len(MODEL_NAME)>2 and MODEL_NAME[:3] in model_map:
    model_type=MODEL_NAME[:3]

print(model_type)
model = model_map[model_type]
model = models.loadModel(MODEL_NAME, model)
weights_only = True if model else False

training.basicTrain(model, epochs=32, n=-1, batch_size=64, learning_rate=0.01, weights_only=weights_only)
models.saveModel(model, model_type+"_"+
                        datetime.now().strftime('%d-%m-%y %H-%M-%S') if SAVE_MODEL else 'temp', weights_only=weights_only)
imgs = training.predict(model, PREDICT_IMG)
training.plot(imgs)

#Unable to restore custom object of type _tf_keras_metric. Please make sure that any custom layers are
# included in the `custom_objects` arg when calling `load_model()` and make sure that all layers implement
# `get_config` and `from_config`.