from project import models
import os
os.chdir("..")

model = models.modelv1((128,128), 5)
models.saveModel(model, 'v1')

