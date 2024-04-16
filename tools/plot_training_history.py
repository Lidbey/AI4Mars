import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
import os


MODEL_NAME = 'URX_resnext50'
df = pd.read_csv(MODEL_NAME + '.csv')

fig, ax = plt.subplots()
fig.set_figwidth(12.8)

ax.plot(df['epoch'], df['categorical_accuracy'], "-o")
ax.plot(df['epoch'], df['loss'], "-o")
ax.plot(df['epoch'], df['mean_absolute_error'], "-o")

ax.legend(['Categorical accuracy', 'Loss', 'Mean absolute error'])
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.grid(True, which="both")

plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title(MODEL_NAME)

if not os.path.exists('history'):
    os.makedirs('history')
plt.savefig(f'history/{MODEL_NAME}.png')

plt.show()
