import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
path = 'labele_train.csv'

df = pd.read_csv(path)

Soil = list(df.ProcSoil)
BedRock = list(df.ProcBedRock)
Sand = list(df.ProcSand)
BigRock = list(df.ProcBigRock)
Null = list(df.ProcNull)

NpSoil = np.array(Soil)
NpBedRock = np.array(BedRock)
NpSand = np.array(Sand)
NpBigRock = np.array(BigRock)
NpNull = np.array(Null)

x = np.arange(0,16064)
xticks = np.arange(0, 16065, 1004)
yticks = np.arange(0, 101, 20)
NpSoil.sort()
NpBedRock.sort()
NpSand.sort()
NpBigRock.sort()
NpNull.sort()


fig, ax = plt.subplots(5, figsize = (10,10))
ax[0].set_title('Soil')
ax[0].plot(NpSoil)
ax[0].set_yticks(yticks)
ax[0].set_xticks(xticks)



ax[1].set_title('BedRock')
ax[1].plot(NpBedRock)
ax[1].set_yticks(yticks)
ax[1].set_xticks(xticks)

ax[2].set_title('Sand')
ax[2].plot(NpSand)
ax[2].set_yticks(yticks)
ax[2].set_xticks(xticks)

ax[3].set_title('BigRock')
ax[3].plot(NpBigRock)
ax[3].set_yticks(yticks)
ax[3].set_xticks(xticks)

ax[4].set_title('Null')
ax[4].plot(NpNull)
ax[4].set_yticks(yticks)
ax[4].set_xticks(xticks)

plt.tight_layout()
plt.show()



