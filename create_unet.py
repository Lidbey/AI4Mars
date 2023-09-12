import unet
import models

model = unet.Unet_resnext50()
models.saveModel(model, 'URX_resnext50', weights_only=True)
