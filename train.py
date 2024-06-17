import os
import pickle
import keras
import base_model 
from sklearn.metrics import confusion_matrix
from datetime import datetime

TRAIN_SET_PATH = os.path.join("..", "data", "train_set.pkl")
TEST_SET_PATH = os.path.join("..", "data", "test_set.pkl")
SAVED_MODELS_PATH = "saved_models"

def test(model, test_set, model_name):
    y_predict = model.predict([test_set['mr'], test_set['rtd'], test_set['clinic_data']])
    y_predict = [1 if y > .5 else 0 for y in y_predict]

    C = confusion_matrix(test_set['label'], y_predict)

    TP = C[1,1] 
    TN = C[0,0]
    FP = C[0,1]
    FN = C[1,0]

    print(C)

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-score: {f1_score:.4f}")
    
    log = f"{model_name}\nAccuracy: {accuracy:.4f}\nSensitivity (Recall): {sensitivity:.4f}\nSpecificity: {specificity:.4f}\nF1-score: {f1_score:.4f}\n\n"
    with open(os.path.join('saved_models','log_results.txt'), "a") as logfile:
        logfile.write(log)

def get_trained_model(load_from_file=False):
    model = None
    if load_from_file:
        last_model = sorted([f for f in os.listdir(SAVED_MODELS_PATH) if f.endswith('.keras')], reverse=True)[0]
        model = keras.saving.load_model(os.path.join(SAVED_MODELS_PATH, last_model))
        model_name = last_model
    else:
        model = base_model.get_model()
        
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit( [train_test['mr'], train_test['rtd'], train_test['clinic_data']], train_test['label'], epochs=25, batch_size=64 )
        
        model_name = f"{datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')}_model.keras"
        
        model.save(os.path.join(SAVED_MODELS_PATH, model_name))
        
    return model, model_name

if __name__ == "__main__":
    with open(TRAIN_SET_PATH, "rb") as f_train, open(TEST_SET_PATH, "rb") as f_test:
        train_test = pickle.load(f_train)
        test_set = pickle.load(f_test)    

    model, model_name = get_trained_model(load_from_file=False)
    
    test(model, test_set, model_name)