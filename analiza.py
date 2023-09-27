from PIL import Image
import os
import pandas as pd

import numpy as np
# import required module
from pathlib import Path
path = 'masked-gold-min3-100agree'
klSoil = []
klBedrock = []
klSand = []
klBigRock = []
klNull = []
klNic = []
klWszystko = []
Nazwy = []

for filename in os.listdir(path):
    if filename.endswith(".png"):
        im = Image.open("masked-gold-min3-100agree/{}".format(filename)).convert('RGB')
        soil = 0
        bedrock = 0
        sand = 0
        bigRock = 0
        null = 0
        nic = 0
        suma = 0

        for pixel in im.getdata():
            if pixel == (0, 0, 0):
                soil += 1
            elif pixel == (1, 1, 1):
                bedrock += 1
            elif pixel == (2, 2, 2):
                sand += 1
            elif pixel == (3, 3, 3):
                bigRock += 1
            elif pixel == (255, 255, 255):
                null += 1
            else:
                nic += 1

        suma = soil + bedrock + sand + bigRock + null + nic
    Nazwy.append(str(filename))
    klSoil.append(soil)
    klBedrock.append(bedrock)
    klSand.append(sand)
    klBigRock.append(bigRock)
    klNull.append(null)
    klNic.append(nic)
    klWszystko.append(suma)

df = pd.DataFrame(list(zip(Nazwy,klSoil,klBedrock,klSand,klBigRock,klNull,klNic,klWszystko)),columns = ['NazwaZdj', 'Soil', 'Bedrock', 'Sand', ' BigRock', 'Null', 'Nic', 'Wszystko'])


df.to_excel('labele_train.xlsx', index = False)




