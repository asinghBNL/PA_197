import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = "./manual_digitized_curves/01_ia_0.csv"

df = pd.read_csv(file)
Ua = df['UA_kV'].to_numpy()
UG1 = df['UG1_V'].to_numpy()

plt.plot(Ua,UG1)
plt.grid()
plt.show()
