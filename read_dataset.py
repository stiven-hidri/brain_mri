import os
import pandas as pd
import numpy as np
import pydicom as dicom
import pydicom
from rt_utils import RTStructBuilder
import logging
import shutil
DATA_DIRECTORY = '../descriptive'
METADATA_PATH = os.path.join(DATA_DIRECTORY, 'metadata.csv')
OUTPUT = '../data'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler() 
logger.addHandler(console_handler)

def resample_rtdose_to_mri(rtdose, mri):
    dose_grid = rtdose.pixel_array
    dose_grid_scaling = rtdose.DoseGridScaling
    dose_origin = rtdose.ImagePositionPatient  # in mm
    dose_orientation = rtdose.ImageOrientationPatient
    dose_pixel_spacing = rtdose.PixelSpacing  # in mm/pixel
    dose_slice_thickness = rtdose.SliceThickness  # in mm

    mri_origin = mri.ImagePositionPatient
    mri_orientation = mri.ImageOrientationPatient
    mri_pixel_spacing = mri.PixelSpacing
    mri_slice_thickness = mri.SliceThickness

    # Construct coordinate arrays for the RTDOSE grid
    dose_shape = dose_grid.shape

    # ... Construct dose_coords: 3D coordinates of the RTDOSE grid points (x, y, z) ...
    # (Implementation depends on the specifics of your DICOM files and coordinate systems.
    # You might need to consider slice orientation, direction cosines, etc.)

    # Calculate the transformation matrix (consider using a specialized library)
    # ... transform_matrix = calculate_transformation(dose_origin, dose_orientation, dose_pixel_spacing, dose_slice_thickness,
    #                                                 mri_origin, mri_orientation, mri_pixel_spacing, mri_slice_thickness)

    # Map dose_coords to the MRI space using the transformation matrix
    mri_coords = transform_matrix @ dose_coords

    # Resample the RTDOSE using interpolation (e.g., linear)
    resampled_dose = map_coordinates(dose_grid * dose_grid_scaling, mri_coords, order=1, mode='nearest')

    # Reshape the resampled dose to match the MRI shape
    resampled_dose = resampled_dose.reshape(mri.pixel_array.shape)
    return resampled_dose

def save_dcm_to_npy():
    metadata_by_subjectid = pd.read_csv(METADATA_PATH).groupby(['Subject ID'])
    print("Saving dcm to npy...")
    for (subject_id, values) in metadata_by_subjectid:
        metadata_by_studyuid = values.groupby(["Study UID"])
        
        subject_id=subject_id[0]
        for (study_uid, values) in metadata_by_studyuid:
            study_uid = study_uid[0]
            dir_name = f"{subject_id}_{study_uid}"
            os.mkdir(os.path.join(OUTPUT, dir_name))
            values.sort_values(by='Modality', inplace=True)
            series_path = ''
            for i, r in values.iterrows():
                path = os.path.join(DATA_DIRECTORY, r['File Location'])
                #MR IMAGE
                if r['Modality'] == 'MR':    
                    series_path = path
                    save_MR_npy(path, dir_name)
                #RADIATION DOSE ON LESION
                elif r['Modality'] == 'RTDOSE' :
                    save_RTD_npy(path, dir_name)
                #LESION MARKED
                elif r['Modality'] == 'RTSTRUCT':
                    save_RTS_npy(path, dir_name, series_path)
                else:
                    logger.debug("This modality shouldn't exist: " + r['Modality'])
            
            crop_lesions(dir_name)      
            
            print(f"\rsave_dcm_to_npy: {subject_id} done", end='')
    
    print("\ndone\n")

def save_MR_npy(files_path: str, dir_name:str):
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
            masks = []
            for name in names:
                if "Skull" not in name: # exclude skull annotations
                    mask_3d = rtstruct.get_roi_mask_by_name(name)
                    masks.append(mask_3d * 1)
            
            combined_mask = np.logical_or.reduce(masks).astype(np.uint8)                      
            np.save(os.path.join(OUTPUT, dir_name,  f"{dir_name}_RTS.npy"), combined_mask)

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
            
def crop_lesions(dir_name:str):   
    MR_path = os.path.join(OUTPUT, dir_name, f"{dir_name}_MR.npy")
    RTS_path = os.path.join(OUTPUT, dir_name, f"{dir_name}_RTS.npy")
    MR_arr = np.load(MR_path)
    LES_mask_arr = np.load(RTS_path)
    LES_arr = LES_mask_arr * MR_arr
    LES_arr_cropped = crop_les(LES_arr)
    np.save(os.path.join(OUTPUT, dir_name, f"{dir_name}_LES.npy"), LES_arr)
    np.save(os.path.join(OUTPUT, dir_name, f"{dir_name}_LESc.npy"), LES_arr_cropped)

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
