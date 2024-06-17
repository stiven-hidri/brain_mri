from fuzzywuzzy import fuzz, process
import os
import shutil
def couple_roi_names(clinical_names, target):
    matches = {}
    if len(clinical_names) == 1 and len(clinical_names) == 1:
        matches[clinical_names[0]] = target[0]
    else:
        for c in clinical_names:
            best_match = process.extractOne(c, target, scorer=fuzz.WRatio)
            if best_match:
                matches[c] = best_match[0]
            
    return matches

def clear_directory_content(PATH):
    for filename in os.listdir(PATH):
        file_path = os.path.join(PATH, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def rm_pss(s:str):
    return '_'.join(c for c in s if c.isalnum())