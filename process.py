import numpy as np
import pandas as pd

from pathlib import Path
directory_ia  = Path("./data/ia/")
directory_ig2 = Path("./data/ig2/")


ia_ua  = np.array([])
ia_ug1 = np.array([])

for file in directory_ia.iterdir():
    if file.is_file():
        temp_file = pd.read_csv(file)
        ia_ua  = np.concatenate((temp_file['UA_kV'].to_numpy(),ia_ua))
        ia_ug1 = np.concatenate((temp_file['UG1_V'].to_numpy(),ia_ug1))

print(ia_ua)
print(ia_ug1)
