import numpy as np
from matplotlib import pyplot as plt
import sys
import os

def display(plt, axes, images, cur_row, c, f_type):
    for i in range(c):
        axes[cur_row, i].imshow(images[i])
        axes[cur_row, i].set_axis_off()

def display_data(folder_path:str):
    files = [f for f in os.listdir(folder_path)]
    
    RTD_file = [f for f in files if f.endswith("RTD.npy")][0]
    MR_file = [f for f in files if f.endswith("MR.npy")][0]
    RTS_file = [f for f in files if f.endswith("RTS.npy")][0]
    MR_les_file = [f for f in files if f.endswith("MR_les.npy")][0]
    RTD_les_file = [f for f in files if f.endswith("RTD_les.npy")][0]
    
    RTD_array = np.load(os.path.join(folder_path, RTD_file))
    MR_array = np.load(os.path.join(folder_path, MR_file))
    RTS_array = np.load(os.path.join(folder_path, RTS_file))
    RTD_les_array = np.load(os.path.join(folder_path, RTD_les_file))
    MR_les_array = np.load(os.path.join(folder_path, MR_les_file))
    r,c = 5, 9

# Create the figure and subplots
    fig, axes = plt.subplots(nrows=r, ncols=c, figsize=(20,10))
    
    buono = np.nonzero(RTS_array>0)
    buono = np.hstack([b for b in buono])
    buono = np.sort(np.unique(buono))
    print(buono)
    
    indexes = np.linspace((np.max(buono)+np.min(buono))//2, np.max(buono), c+2, dtype=int)[1:-1]
    
    print(indexes)
    
    images = RTS_array[indexes]
    display(plt, axes, images, 0, c, "RTS")
    
    images = RTD_array[indexes]
    display(plt, axes, images, 1, c, "RTD")
    
    images = MR_array[indexes]
    display(plt, axes, images, 2, c, "MR")
    
    images = MR_les_array[indexes]
    display(plt, axes, images, 3, c, "MR")
    
    images = RTD_les_array[indexes]
    display(plt, axes, images, 4, c, "MR")
    
    os.makedirs(os.path.join(os.curdir, "sample_images"), exist_ok=True)
    name = folder_path.split('\\')[-1]
    plt.savefig(os.path.join(os.curdir, "sample_images", f"{name.replace('.','')}.png"))
        
    # plt.figure(1)
    # plt.imshow(RTD_array[0], cmap='gray')
    # plt.show()
    # plt.figure(2)
    # plt.imshow(MR_array[200], cmap='gray')
    # plt.show()

if __name__ == "__main__":
    folder_path = sys.argv[1] if len(sys.argv)>1 else r"C:\\Users\\hidri\\TESI\\data\\GK_103_1.3.6.1.4.1.14519.5.2.1.261238491105529422607835392969394449648"
    display_data(folder_path)