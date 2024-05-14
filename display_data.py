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
    
    RTD_file = [f for f in files if "RTD" in f][0]
    MR_file = [f for f in files if "MR" in f][0]
    RTS_file = [f for f in files if "RTS" in f][1]
    
    RTD_array = np.load(os.path.join(folder_path, RTD_file))
    MR_array = np.load(os.path.join(folder_path, MR_file))
    RTS_array = np.load(os.path.join(folder_path, RTS_file))
    
    r,c = 3, 9

# Create the figure and subplots
    fig, axes = plt.subplots(nrows=r, ncols=c, figsize=(20,10))
    
    images = [RTS_array[i] for i in np.linspace(0, RTS_array.shape[0], c+2, dtype=int)[1:-1]]
    display(plt, axes, images, 0, c, "RTS")
    
    images = [RTD_array[i] for i in np.linspace(0, RTD_array.shape[0], c+2, dtype=int)[1:-1]]
    display(plt, axes, images, 1, c, "RTD")
    
    images = [MR_array[i] for i in np.linspace(0, MR_array.shape[0], c+2, dtype=int)[1:-1]]
    display(plt, axes, images, 2, c, "MR")
    
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