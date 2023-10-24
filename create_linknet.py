import models

model = models.Linknet_densenet201()
models.saveModel(model, 'LNT_densenet201', weights_only=True)
