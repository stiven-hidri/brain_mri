import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from rt_utils import RTStructBuilder
from utils import couple_roi_names
import SimpleITK as sitk

def handle_mr_rtd_sitk(path_MR, path_RTD):
    
    reader_mr = sitk.ImageSeriesReader()
    reader_mr_names = reader_mr.GetGDCMSeriesFileNames(path_MR)
    reader_mr.SetFileNames(reader_mr_names)
    mr = reader_mr.Execute()
    
    if mr.GetDimension()==4 and mr.GetSize()[3]==1:
        mr = mr[...,0]
    
    reader_rtd = sitk.ImageSeriesReader()
    reader_rtd_names = reader_rtd.GetGDCMSeriesFileNames(path_RTD)
    reader_rtd.SetFileNames(reader_rtd_names)
    rtd = reader_rtd.Execute()

    if rtd.GetDimension()==4 and rtd.GetSize()[3]==1:
        rtd = rtd[...,0]

    rtd = sitk.Resample(rtd, mr)
    
    mr, rtd = sitk.GetArrayFromImage(mr), sitk.GetArrayFromImage(rtd)
    
    return mr, rtd

def __handle_rts__( path_RTS, series_path):
    rt_struct_path = [os.path.join(path_RTS, f) for f in os.listdir(path_RTS) if f.endswith('.dcm')][0]
    rtstruct = RTStructBuilder.create_from(dicom_series_path=series_path, rt_struct_path=rt_struct_path)
    
    final = None
    
    for i, roi in enumerate(rtstruct.get_roi_names()):
        if 'Skull' not in roi: # exclude skull annotations
            mask_3d = rtstruct.get_roi_mask_by_name(roi)
            mask_3d = mask_3d * 1
            mask_3d = np.swapaxes(mask_3d, 0, 2)
            mask_3d = np.swapaxes(mask_3d, 1, 2)
            if(i == 0):
                final = mask_3d
            else:
                final = np.logical_or(final, mask_3d)
                
    return final

def plot(stuff, c):
    r = len(stuff)
    
    fig, axes = plt.subplots(nrows=r, ncols=c, figsize=(12,6))
    indexes = []
    
    for i in range(stuff[2].shape[0]):
        if(np.sum(stuff[2][i].reshape((1,-1))) > 0):
            indexes.append(i)
            
    np.linspace(min(indexes), max(indexes), c+2, dtype=int)[1:-1]
    
    for i, s in enumerate(stuff):
        images = s[indexes]
        display(plt, axes, images, i, c)
    
    os.makedirs(os.path.join(os.curdir, "sample_images"), exist_ok=True)
    plt.savefig(os.path.join(os.curdir, "sample_images", f"test_affine.png"))
    plt.show()

def display(plt, axes, images, cur_row, c):
    for i in range(c):
        axes[cur_row, i].imshow(images[i])
        axes[cur_row, i].set_axis_off()

if __name__ == "__main__":
    path_MR = os.path.join('test_data', 'mr')
    path_RTD = os.path.join('test_data', 'rtd')
    path_RTS = os.path.join('test_data', 'rts')
    mr, rtd = handle_mr_rtd_sitk(path_MR, path_RTD)
    rts = __handle_rts__(path_RTS, path_MR)
    
    print(rts.shape)
    print(mr.shape)
    print(rtd.shape)
    
    plot([mr, rtd, rts], 12)