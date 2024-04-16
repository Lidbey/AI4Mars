from project import models
import os
os.chdir("..")

model = models.Unet_resnext50()
models.saveModel(model, 'URX_resnext50', weights_only=True)
