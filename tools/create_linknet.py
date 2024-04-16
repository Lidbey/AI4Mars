from project import models
import os
os.chdir("..")

model = models.Linknet_densenet201()
models.saveModel(model, 'LNT_densenet201', weights_only=True)
