import segmentation_models as sm
from keras.layers import Input, Conv2D
from keras.models import Model

def Unet_resnext50():
    BACKBONE = 'resnext50'
    preprocess_input = sm.get_preprocessing(BACKBONE)

    base_model = sm.Unet(BACKBONE, classes=5, activation='softmax', encoder_weights='imagenet')

    inp = Input(shape=(None, None, 1))
    l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
    out = base_model(l1)

    model = Model(inp, out, name=base_model.name)

    model.compile(
        'Adam',
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )
    return model



