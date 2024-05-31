import os
import pandas as pd
import numpy as np
import pydicom as dicom
import pydicom
from rt_utils import RTStructBuilder
import shutil
from scipy.ndimage import zoom
from sklearn import preprocessing
from utils import rm_pss, couple_roi_names

class Data_Reader():
    def __init__(self) -> None:
        #paths
        self.RAW_DATA_PATH = os.path.join('..', 'descriptive')
        self.RAWDATA_META_PATH = os.path.join(self.RAW_DATA_PATH, 'metadata.csv')
        self.CLINIC_DATA_PATH = os.path.join('..', 'Brain-TR-GammaKnife-Clinical-Information.xlsx')
        self.OUTPUT_DATA_PATH = os.path.join('..', 'data')
        self.ALL_CLINIC_DATA_PATH =  os.path.join(self.OUTPUT_DATA_PATH, 'all_clinic_data.npy')
        self.ALL_LESIONS_PATH =  os.path.join(self.OUTPUT_DATA_PATH, 'all_lesions.npy')
        self.ALL_RTDOSES_PATH =  os.path.join(self.OUTPUT_DATA_PATH, 'all_rtdoses.npy')
        self.ALL_LABELS_PATH =  os.path.join(self.OUTPUT_DATA_PATH, 'all_labels.npy')

        #adjusting metadata and clinical data dataframes
        self.rawdata_meta = pd.read_csv(self.RAWDATA_META_PATH)
        self.clinic_data_ll = pd.read_excel(self.CLINIC_DATA_PATH, sheet_name='lesion_level')
        self.clinic_data_cl = pd.read_excel(self.CLINIC_DATA_PATH, sheet_name='course_level')
        
        self.clinic_data = None
        
        #final outputs 
        self.all_clinic_data = []
        self.all_rtdoses = []
        self.all_lesions = []
        self.all_labels = []
        
    def __adjust_dataframes__(self):
        rawdata_meta_renaming = {'Study Date':'study_date','Study UID':'study_uid','Subject ID':'subject_id', 'Modality':'modality', 'File Location':'file_path'}
        clinic_data_ll_renaming = {'unique_pt_id': 'subject_id', 'Treatment Course':'course', 'Lesion Location':'roi', 'mri_type':'label', 'duration_tx_to_imag (months)': 'duration_tx_to_imag', 'Fractions':'fractions'}
        clinic_data_cl_renaming = {'unique_pt_id': 'subject_id', 'Course #':'course', 'Diagnosis (Only want Mets)':'mets_diagnosis', 'Primary Diagnosis':'primary_diagnosis', 'Age at Diagnosis':'age', 'Gender':'gender'}
        
        rawdata_meta_types = {'study_date':"datetime64[ns]" ,'study_uid':'string','subject_id':'string', 'modality':'string', 'file_path':'string'}
        clinic_data_ll_types = {'subject_id':'int64','course':'Int8', 'roi':'string', 'label':'string', 'duration_tx_to_imag':'int8', 'fractions':'int8'}
        clinic_data_cl_types = {'subject_id':'int64', 'course':'Int8', 'mets_diagnosis':'string', 'primary_diagnosis':'string', 'age':'int8', 'gender':'string'}
        
        self.rawdata_meta = self.rawdata_meta.drop(self.rawdata_meta.columns.difference(rawdata_meta_renaming.keys()), axis=1).rename(columns=rawdata_meta_renaming)
        self.clinic_data_ll = self.clinic_data_ll.drop(self.clinic_data_ll.columns.difference(clinic_data_ll_renaming.keys()), axis=1).rename(columns=clinic_data_ll_renaming)
        self.clinic_data_cl = self.clinic_data_cl.drop(self.clinic_data_cl.columns.difference(clinic_data_cl_renaming.keys()), axis=1).rename(columns=clinic_data_cl_renaming)
        
        self.rawdata_meta = self.rawdata_meta.astype(rawdata_meta_types)
        self.clinic_data_ll = self.clinic_data_ll.astype(clinic_data_ll_types)
        self.clinic_data_cl = self.clinic_data_cl.astype(clinic_data_cl_types)
        
        self.clinic_data = pd.merge_ordered(self.clinic_data_ll, self.clinic_data_cl, on=['subject_id', 'course'], how='inner')

    def __clean_output_directory__(self):
        for filename in os.listdir(self.OUTPUT_DATA_PATH):
            file_path = os.path.join(self.OUTPUT_DATA_PATH, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    def __get_mr_rtd_rts_path__(self, values):
        path_MR = os.path.join(self.RAW_DATA_PATH, values.loc[values['modality'] == 'MR', 'file_path'].iloc[0])
        path_RTD = os.path.join(self.RAW_DATA_PATH, values.loc[values['modality'] == 'RTDOSE', 'file_path'].iloc[0])
        path_RTS = os.path.join(self.RAW_DATA_PATH, values.loc[values['modality'] == 'RTSTRUCT', 'file_path'].iloc[0])
        
        return path_MR, path_RTD, path_RTS

# TODO: check correct preprocessing (armonic stuff)
    def __handle_mr_rtd__(self, path_MR, path_RTD): 
        #MR dicoms
        file_names_MR = [os.path.join(path_MR, f) for f in os.listdir(path_MR) if f.endswith('.dcm')]        
        mr_data = [pydicom.dcmread(p) for p in file_names_MR] 
        
        mr_pixel_data = [p.pixel_array for p in mr_data] 
        
        RefDs = mr_data[0]
        
        ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(file_names_MR))
        ArrayDicom =  np.empty(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
        for i, pa in enumerate(mr_pixel_data):
            ArrayDicom[:, :, i] = pa
        
        #RTD dicom  
        file_path_RTD  = [os.path.join(path_RTD, f) for f in os.listdir(path_RTD) if f.endswith('.dcm')][0]
        dcm_RTD = pydicom.dcmread(file_path_RTD)
        rtdose_array = dcm_RTD.pixel_array
        
        target_shape = ArrayDicom.shape
        original_shape = rtdose_array.shape
        factor = (target_shape[0]/original_shape[0], target_shape[1]/original_shape[1], target_shape[2]/original_shape[2])
        
        mapped_dose = zoom(rtdose_array, factor)
        
        return ArrayDicom, mapped_dose

    def __handle_rts__(self, path_RTS, series_path, subject_id, course):
        rt_struct_path = [os.path.join(path_RTS, f) for f in os.listdir(path_RTS) if f.endswith('.dcm')][0]
        rtstruct = RTStructBuilder.create_from(dicom_series_path=series_path, rt_struct_path=rt_struct_path)
        rois = self.clinic_data.loc[(self.clinic_data['subject_id'] == subject_id) & (self.clinic_data['course'] == course), 'roi'].values
        
        couples = couple_roi_names(rois, rtstruct.get_roi_names())
        
        masks = {}
        for roi in rois:
            if 'Skull' not in roi: # exclude skull annotations
                mask_3d = rtstruct.get_roi_mask_by_name(couples[roi])
                masks[roi] = mask_3d * 1
                
        return masks

    def __get_labels__(self, rois, subject_id, course):
        to_return = {}
        
        for roi in rois:    
            label = self.clinic_data.loc[(self.clinic_data['subject_id']==subject_id)&(self.clinic_data['course']==course)&(self.clinic_data['roi']==roi), ['label']].values[0]
            to_return[roi] = label
            
        return to_return

    def __get_clinic_data__(self, rois, subject_id, course):
        to_return = {}
        
        for roi in rois:    
            clinic_data_row = self.clinic_data.loc[(self.clinic_data['subject_id']==subject_id)&(self.clinic_data['course']==course)&(self.clinic_data['roi']==roi), ['mets_diagnosis', 'primary_diagnosis', 'age', 'gender', 'duration_tx_to_imag', 'fractions']].values[0]
            to_return[roi] = clinic_data_row
            
        return to_return
    
    def __preprocess__(self, mr, rtd, masks):   
        mr_rtd_roi = self.__mask_and_crop__(mr, rtd, masks)
        
        mr_rtd_roi = self.__resize__(mr_rtd_roi)
        mr_rtd_roi = self.__normalize__(mr_rtd_roi)
        return mr_rtd_roi
    def __mask_and_crop__(self, mr, rtd, masks):
        mr_rtd_roi = {}
        
        for roi in masks.keys():
            mr_masked = masks[roi] * mr
            rtd_masked = masks[roi] * rtd
            
            mr_masked_cropped = self.__crop_les__(mr_masked)
            rtd_masked_cropped = self.__crop_les__(rtd_masked)
            
            mr_rtd_roi[roi] = [mr_masked_cropped, rtd_masked_cropped]
        
        return mr_rtd_roi
    
    def __crop_les__(self, image):
        true_points = np.argwhere(image)
        top_left = true_points.min(axis=0)
        bottom_right = true_points.max(axis=0)
        cropped_arr = image[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1, top_left[2]:bottom_right[2]+1]
        return cropped_arr
        
    def __resize__(Self, mr_rtd_roi, target_shape=40.):
        for roi in mr_rtd_roi.keys():
            mr, rtd = mr_rtd_roi[roi]
            
            factors_mr = (target_shape/float(mr.shape[0]), target_shape/float(mr.shape[1]), target_shape/float(mr.shape[2]))
            factors_rtd = (target_shape/float(rtd.shape[0]), target_shape/float(rtd.shape[1]), target_shape/float(rtd.shape[2]))
            
            mr = zoom(mr, factors_mr)
            rtd = zoom(rtd, factors_rtd)
            
            mr_rtd_roi[roi] = [mr, rtd]
        
        return mr_rtd_roi

    def __normalize__(Self, mr_rtd_roi):
        for roi in mr_rtd_roi.keys():
            mr, rtd = mr_rtd_roi[roi]
            
            mr = np.float64(mr) 
            rtd = np.float64(rtd)
            
            mr /= 255.
            rtd /= 255.
            
            mr_rtd_roi[roi] = [mr, rtd]
        
        return mr_rtd_roi

    def __augment__(self, label_roi, mr_rtd_roi, clinic_data_roi):
        rois = mr_rtd_roi.keys()
        for roi in rois:
            if label_roi[roi] != 'stable':
                mr, rtd = mr_rtd_roi[roi]
                augmented_mr = self.__rotate_image__(mr,roi)
                augmented_rtd = self.__rotate_image__(rtd,roi)
                
                augmented_mr_rtd = {n_roi:[augmented_mr[n_roi], augmented_rtd[n_roi]] for n_roi in augmented_mr.keys()}
                
                new_rois = augmented_mr.keys()
                
                augmented_label = {n_roi:label_roi[roi] for n_roi in new_rois}
                augmented_clinic_data = {n_roi:clinic_data_roi[roi] for n_roi in new_rois}
                
                label_roi |= augmented_label
                mr_rtd_roi |= augmented_mr_rtd
                clinic_data_roi |= augmented_clinic_data
                
        return label_roi, mr_rtd_roi, clinic_data_roi 

    def __rotate_image__(self, image, roi) -> dict:
        rotated_images = {}
        axes = {'z':(0,1), 'x':(1, 2), 'y':(0,2)}
        angles = [90, 180, 270]
        
        for axes_key in axes.keys():
            for angle in angles:
                k = angle // 90  # Number of 90-degree rotations
                rotated_images[f'{roi}_{axes_key}_{angle}'] = np.rot90(image, k, axes[axes_key])
        
        return rotated_images
    
    def __append_to_output_lists__(self, label_roi, mr_rtd_roi, clinic_data_roi):
        for roi in mr_rtd_roi.keys():
            mr, rtd = mr_rtd_roi[roi]
            label = label_roi[roi]
            clinic_data = clinic_data_roi[roi]

        
            self.all_clinic_data.append(clinic_data)
            self.all_lesions.append(mr)
            self.all_rtdoses.append(rtd)
            self.all_labels.append(label)
    
    def __one_hot__(self):
        self.all_labels = 1 - preprocessing.LabelEncoder().fit(self.all_labels).transform(self.all_labels)
        
        for j in [0, 1, 3]:
            self.all_clinic_data = np.array(self.all_clinic_data)
            self.all_clinic_data[:,j] = preprocessing.LabelEncoder().fit(self.all_clinic_data[:,j]).transform(self.all_clinic_data[:,j])

        self.all_clinic_data = [[int(float(x)) for x in row] for row in self.all_clinic_data]
    
    def __save__(self):
        np.save(self.ALL_LABELS_PATH, np.array(self.all_labels))
        np.save(self.ALL_CLINIC_DATA_PATH, self.all_clinic_data)
        np.save(self.ALL_LESIONS_PATH,  np.array(self.all_lesions))
        np.save(self.ALL_RTDOSES_PATH, np.array(self.all_rtdoses))
    
    def main(self, cleanOutDir = True):
        self.__adjust_dataframes__()
        
        if cleanOutDir:
            self.__clean_output_directory__()
        
        metadata_by_subjectid = self.rawdata_meta.groupby(['subject_id'])
        total_subjects = len(metadata_by_subjectid)
        
        print('Saving dcm to npy...')
        
        for cnt, (subject_id, values) in enumerate(metadata_by_subjectid):
            metadata_by_studyuid = [(k,v) for (k,v) in values.groupby(['study_uid'])]
            metadata_by_studyuid = sorted(metadata_by_studyuid, key = lambda x: x[1]['study_date'].iloc[0])
            
            subject_id = int(subject_id[0].split('_')[1]) # remove GK
            
            for course ,(_, values) in enumerate(metadata_by_studyuid, start=1):
                path_MR, path_RTD, path_RTS = self.__get_mr_rtd_rts_path__(values)
                
                mr, rtd = self.__handle_mr_rtd__(path_MR, path_RTD)        
                masks = self.__handle_rts__(path_RTS, path_MR, subject_id, course)
                
                mr_rtd_roi = self.__preprocess__(mr, rtd, masks)
                labels_roi = self.__get_labels__(masks.keys(), subject_id, course)
                clinic_data_roi = self.__get_clinic_data__(masks.keys(), subject_id, course)
                
                labels_roi, mr_rtd_roi, clinic_data_roi = self.__augment__(labels_roi, mr_rtd_roi, clinic_data_roi)
                
                self.__append_to_output_lists__(labels_roi, mr_rtd_roi, clinic_data_roi)
                    
            print(f'\rStep: {cnt+1}/{total_subjects}', end='')
        
        self.__one_hot__()
        self.__save__()
        
        print('Data has been read successfully!\n')

if __name__ == '__main__':
    dr = Data_Reader()
    dr.main(cleanOutDir=True)