from datetime import datetime
import models
from stats import calc_stats


MODEL_NAME = 'URX_resnext50'
model_map={"URX": models.Unet_resnext50, "LNT": models.Linknet_densenet201, "DFT": models.default}

model_type = "DFT"
if len(MODEL_NAME)>2 and MODEL_NAME[:3] in model_map:
    model_type=MODEL_NAME[:3]

print(model_type)
model = model_map[model_type]()
weights_only = True if model else False
model = models.loadModel(MODEL_NAME, model)

save_datetime = datetime.now().strftime('%d-%m-%y %H-%M-%S')

calc_stats(model, MODEL_NAME, save_datetime)
