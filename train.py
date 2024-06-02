import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from tensorflow import keras
from keras import optimizers, layers, metrics
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from model import get_model
import os
import pickle

## TODO: aggiusta sto train diobono

TRAIN_PATH = os.path.join("..", "data", "train_set.pkl")
TEST_PATH = os.path.join("..", "data", "test_set.pkl")
with open(TRAIN_PATH, "rb") as f:
    train_test = pickle.load(f)
with open(TEST_PATH, "rb") as f:
    test_set = pickle.load(f)    

model = get_model()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit( [train_test['mr'], train_test['rtd'], train_test['clinic_data']], train_test['label'], epochs=25, batch_size=64 )
model.save("model.keras")

#model = keras.models.load_model("model.h5")

y_predict = model.predict([test_set['mr'], test_set['rtd'], test_set['clinic_data']])
y_predict = [1 if y > .5 else 0 for y in y_predict]

from sklearn.metrics import confusion_matrix
C = confusion_matrix(test_set['label'], y_predict)

TP = C[1,1] 
TN = C[0,0]
FP = C[0,1]
FN = C[1,0]

accuracy = (TP + TN) / (TP + TN + FP + FN)
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
precision = TP / (TP + FP)
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1-score: {f1_score:.4f}")
