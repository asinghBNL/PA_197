import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
directory_ia  = Path("./data/28/ia/")
directory_ig2 = Path("./data/28/ig2/")

ia_ua  = []
ia_ug1 = []
for file in sorted(directory_ia.iterdir(), key=lambda x: x.name):
    if file.is_file():
        print(file.name)
        temp_file = pd.read_csv(file)
        ia_ua.append(1e3*temp_file['UA_kV'].to_numpy())
        ia_ug1.append(temp_file['UG1_V'].to_numpy())

ig2_ua  = []
ig2_ug1 = []
for file in sorted(directory_ig2.iterdir(), key=lambda x: x.name):
    if file.is_file():
        print(file.name)
        temp_file = pd.read_csv(file)
        ig2_ua.append(1e3*temp_file['UA_kV'].to_numpy())
        ig2_ug1.append(temp_file['UG1_V'].to_numpy())

# default voltage grid
# Ua_domain  = np.arange(0e3,12e3,10)
Ua_domain  = np.arange(0e3,20e3,10)

ia_ug1_interp = ia_ug1.copy()
for i in range(len(ia_ug1)):
    ia_ug1_interp[i] = np.interp(Ua_domain,ia_ua[i],ia_ug1[i],left=np.nan,right=np.nan)

ig2_ug1_interp = ig2_ug1.copy()
for i in range(len(ig2_ug1)):
    ig2_ug1_interp[i] = np.interp(Ua_domain,ig2_ua[i],ig2_ug1[i],left=np.nan,right=np.nan)

for i in ia_ug1_interp:
    plt.plot(Ua_domain,i,color='red')

for i in ig2_ug1_interp:
    plt.plot(Ua_domain,i,color='blue')

plt.grid()
plt.show()

ia_arr = [Ua_domain,*ia_ug1_interp]
np.savetxt("ia_arr.csv",ia_arr,delimiter=',',fmt='%.10e')
ig2_arr = [Ua_domain,*ig2_ug1_interp]
np.savetxt("ig2_arr.csv",ig2_arr,delimiter=',',fmt='%.10e')
