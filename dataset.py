import os
import pandas as pd
import os
import numpy as np
import pydicom as dicom
from rt_utils import RTStructBuilder
#import SimpleITK as sitk
import logging
import pydicom
import shutil

DATA_DIRECTORY = '../descriptive'
METADATA_PATH = os.path.join(DATA_DIRECTORY, 'metadata.csv')
OUTPUT = '../data'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler() 
logger.addHandler(console_handler)

def save_dcm_to_npy():
    metadata_by_key = pd.read_csv(METADATA_PATH).groupby(['Subject ID', 'Study UID'])
    print("\rsave_dcm_to_npy: 0%", end='')
    for i, (keys, values) in enumerate(metadata_by_key):
        subject_id, study_id = keys
        dir_name = f"{subject_id}_{study_id}"
        os.mkdir(os.path.join(OUTPUT, dir_name))
        values.sort_values(by='Modality', inplace=True)
        series_path = ''
        for i, r in values.iterrows():
            path = os.path.join(DATA_DIRECTORY, r['File Location'])
            if r['Modality'] == 'MR':    
                series_path = path
                save_MR_npy(path, dir_name)
            elif r['Modality'] == 'RTDOSE' :
                save_RTD_npy(path, dir_name)
            elif r['Modality'] == 'RTSTRUCT':
                save_RTS_npy(path, dir_name, series_path)
            else:
                logger.debug("This modality shouldn't exist: " + r['Modality'])
        
        print(f"\rsave_dcm_to_npy: {(i+1)//len(metadata_by_key)*100} %", end='')
    
    print("\ndone\n")

def save_MR_npy(files_path: str, dir_name):
    list_files_DCM = []
    for file_name in os.listdir(files_path):
        if file_name.endswith('.dcm'):
            list_files_DCM.append(os.path.join(files_path, file_name))
            RefDs = dicom.read_file(list_files_DCM[0])
            ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(list_files_DCM))
            ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
            for filenameDCM in list_files_DCM:
                ds = dicom.read_file(filenameDCM)
                ArrayDicom[:, :, list_files_DCM.index(filenameDCM)] = ds.pixel_array
            np.save(os.path.join(OUTPUT, dir_name, f"{dir_name}_MR.npy"), ArrayDicom)
            
def save_RTS_npy(file_path:str, dir_name, series_path):
    for file_name in os.listdir(file_path):
        if file_name.endswith('.dcm'):
            rt_struct_path = os.path.join(file_path, file_name)
            rtstruct = RTStructBuilder.create_from(dicom_series_path=series_path, rt_struct_path=rt_struct_path)
            names = rtstruct.get_roi_names()
            for name in names:
                if "Skull" not in name: # exclude skull annotations
                    mask_3d = rtstruct.get_roi_mask_by_name(name)
                    mask = mask_3d * 1
                    np.save(os.path.join(OUTPUT, dir_name,  f"{dir_name}_{name}_RTS.npy"), mask)

def save_RTD_npy(file_path: str, dir_name):
    for file_name in os.listdir(file_path):
        if file_name.endswith('.dcm'):
            rtdose_struct_path = os.path.join(file_path, file_name) 
            ds = pydicom.dcmread(rtdose_struct_path)

            dose_grid = ds.pixel_array * ds.DoseGridScaling  # Get dose data with scaling
            # dose_spacing = [float(x) for x in ds.PixelSpacing] + [float(ds.SliceThickness)]  # X, Y, Z spacing
            # image_position_patient = ds.ImagePositionPatient  # Optional: Image position

            # *** Step 3: Create a NumPy array ***
            rtdose_array = np.array(dose_grid)

            # If necessary, reshape the array based on the number of frames:
            if rtdose_array.ndim == 4:  # Check if there are multiple frames
                num_frames = ds.NumberOfFrames 
                rtdose_array = rtdose_array.reshape(num_frames, *rtdose_array.shape[1:])  
                
            np.save(os.path.join(OUTPUT, dir_name, f"{dir_name}_RTD.npy"), rtdose_array)
            
def crop_lesions():
    dir_names = os.listdir(OUTPUT)
    for i, dir_name in enumerate(dir_names):
        mri_path = os.path.join(OUTPUT, dir_name, f"{dir_name}_MR.npy")
        mask_path = os.path.join(OUTPUT, dir_name, f"{dir_name}_RTS.npy")
        mri_arr = np.load(mri_path)
        les_mask_arr = np.load(mask_path)
        les_arr = les_mask_arr * mri_arr
        c_les_arr = crop_les(les_arr)
        print(c_les_arr.shape)
        np.save(os.path.join(OUTPUT, dir_name, f"{dir_name}_LESION.npy"), les_arr)
        
        print(f"\rcrop_lesions: {i//len(dir_names)*100} %", end='')
    
    print("\ndone\n")

def crop_les(d):
    true_points = np.argwhere(d)
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)
    cropped_arr = d[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1, top_left[2]:bottom_right[2]+1]
    return cropped_arr

def clean_output_directory():
    for filename in os.listdir(OUTPUT):
        file_path = os.path.join(OUTPUT, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Recursively remove subfolders
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

if __name__ == "__main__":
    clean_output_directory()
    save_dcm_to_npy()
    crop_lesions()
