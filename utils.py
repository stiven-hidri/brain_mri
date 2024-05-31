from fuzzywuzzy import fuzz, process

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

def rm_pss(s:str):
    return ''.join(c for c in s if c.isalnum())