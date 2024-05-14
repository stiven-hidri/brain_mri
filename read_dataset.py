import os
import pandas as pd
import numpy as np
import pydicom as dicom
import pydicom
from rt_utils import RTStructBuilder
import logging
import shutil
import skimage
from scipy.ndimage import zoom


DATA_DIRECTORY = os.path.join('..', 'descriptive')
METADATA_PATH = os.path.join(DATA_DIRECTORY, 'metadata.csv')
OUTPUT = os.path.join('..', 'data')

def handle_MR_RTD(path_MR, path_RTD, dir_name): 
    #MR dicoms
    file_names_MR = [os.path.join(path_MR, f) for f in os.listdir(path_MR) if f.endswith('.dcm')]        
    mr_data = [pydicom.dcmread(p) for p in file_names_MR] 
    mr_pixel_data = [p.pixel_array for p in mr_data] 
    
    RefDs = mr_data[0]
    
    # print(f"MR io: {RefDs.ImageOrientationPatient}") 
    
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(file_names_MR))
    ArrayDicom =  np.empty(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
    for i, pa in enumerate(mr_pixel_data):
        ArrayDicom[:, :, i] = pa
    
    
    #RTD dicom  
    file_path_RTD  = [os.path.join(path_RTD, f) for f in os.listdir(path_RTD) if f.endswith('.dcm')][0]
    rtdose_array = pydicom.dcmread(file_path_RTD).pixel_array
    
    # print(f"RTD io: {pydicom.dcmread(file_path_RTD).ImageOrientationPatient}")
    
    target_shape = ArrayDicom.shape
    original_shape = rtdose_array.shape
    factor = (target_shape[0]/original_shape[0], target_shape[1]/original_shape[1], target_shape[2]/original_shape[2])
    
    mapped_dose = zoom(rtdose_array, factor)
    
    # print(f"rtdose_data shape: {rtdose_array.shape}") 
    # print(f"mapped_dose shape: {mapped_dose.shape}")
    # print(f"array dicom shape: {ArrayDicom.shape}")
    
    np.save(os.path.join(OUTPUT, dir_name, f"{dir_name}_RTD.npy"), mapped_dose) 
    np.save(os.path.join(OUTPUT, dir_name, f"{dir_name}_MR.npy"), ArrayDicom)
    
    return (ArrayDicom, mapped_dose)

def save_dcm_to_npy():
    metadata_by_subjectid = pd.read_csv(METADATA_PATH).groupby(['Subject ID'])
    count = 0
    print("Saving dcm to npy...")
    print(f"\r0/76", end='')
    for (subject_id, values) in metadata_by_subjectid:
        metadata_by_studyuid = values.groupby(["Study UID"])
        subject_id=subject_id[0]
        for (study_uid, values) in metadata_by_studyuid:
            study_uid = study_uid[0]
            dir_name = f"{subject_id}_{study_uid}"
            os.makedirs(os.path.join(OUTPUT, dir_name), exist_ok=True)
            values.sort_values(by='Study Date', inplace=True)
            
            path_MR = os.path.join(DATA_DIRECTORY, values.loc[values["Modality"] == "MR", "File Location"].iloc[0])
            path_RTD = os.path.join(DATA_DIRECTORY, values.loc[values["Modality"] == "RTDOSE", "File Location"].iloc[0])
            path_RTS = os.path.join(DATA_DIRECTORY, values.loc[values["Modality"] == "RTSTRUCT", "File Location"].iloc[0])
                        
            (MR, RTD) = handle_MR_RTD(path_MR, path_RTD, dir_name)
            
            RTS = handle_RTS_npy(path_RTS, dir_name, path_MR)
            
            crop_and_save(MR, RTD, RTS, dir_name)      
            
            count +=1
            
            print(f"\r{count}/76", end='')
    
    print("\ndone\n")

def handle_RTS_npy(path_RTS, dir_name, series_path):
    rt_struct_path = [os.path.join(path_RTS, f) for f in os.listdir(path_RTS) if f.endswith('.dcm')][0]
    rtstruct = RTStructBuilder.create_from(dicom_series_path=series_path, rt_struct_path=rt_struct_path)
    names = rtstruct.get_roi_names()
    masks = []
    for name in names:
        if "Skull" not in name: # exclude skull annotations
            mask_3d = rtstruct.get_roi_mask_by_name(name)
            masks.append(mask_3d * 1)
    
    combined_mask = np.logical_or.reduce(masks).astype(np.uint8)
    np.save(os.path.join(OUTPUT, dir_name, f"{dir_name}_RTS.npy"), combined_mask)
    return combined_mask
            
def crop_and_save(MR, RTD, RTS, dir_name):   
    MR_les = RTS * MR
    MR_les_cropped = crop_les(MR_les)
    RTD_les = RTS * RTD
    RTD_les_cropped = crop_les(RTD_les)
    np.save(os.path.join(OUTPUT, dir_name, f"{dir_name}_MR_les.npy"), MR_les)
    np.save(os.path.join(OUTPUT, dir_name, f"{dir_name}_MR_lesc.npy"), MR_les_cropped)
    np.save(os.path.join(OUTPUT, dir_name, f"{dir_name}_RTD_les.npy"), RTD_les)
    np.save(os.path.join(OUTPUT, dir_name, f"{dir_name}_RTD_lesc.npy"), RTD_les_cropped)

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
