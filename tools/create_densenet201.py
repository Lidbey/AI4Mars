from project import models
import os
os.chdir("..")

model = models.modelDN201()
models.saveModel(model, 'DNM_densenet201', weights_only=True)
