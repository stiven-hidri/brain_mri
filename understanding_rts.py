import os
import numpy as np

path = r"C:\\Users\\hidri\\TESI\\data\\GK_103_1.3.6.1.4.1.14519.5.2.1.261238491105529422607835392969394449648"
folder_name = "GK_103_1.3.6.1.4.1.14519.5.2.1.261238491105529422607835392969394449648"

names = [ f.split('_')[-1] for f in os.listdir(path)] 
data = { n:np.load(os.path.join(path, f"{folder_name}_{n}")) for n in names}

for n in names:
    print(f"{n}: {data[n].shape}")