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
from tensorflow.keras import backend as K
from model import get_model
import os

## TODO: aggiusta sto train diobono

RTDOSES_PATH = os.path.join("..", "all_rtdoses.npy")
LESIONS_PATH = os.path.join("..", "all_lesions.npy")
LABELS_PATH = os.path.join("..", "all_labels.npy")
CLINIC_DATA_PATH = os.path.join("..", "all_clinic_data.npy")

all_rtdoses = np.load(RTDOSES_PATH)
all_lesions = np.load(LESIONS_PATH)
all_labels = np.load(LABELS_PATH)
all_clinic_data = np.load(CLINIC_DATA_PATH)

assert len(all_rtdoses) == len(all_lesions) == len(all_labels) == len(all_clinic_data)



def specificity(y_true, y_pred):
    y_pred = tf.math.round(y_pred)  # Round probabilities to 0 or 1 if needed
    tn = tf.math.reduce_sum(tf.cast((1. - tf.cast(y_true, tf.float32)) * (1 - y_pred), tf.float32))
    fp = tf.math.reduce_sum(tf.cast((1. - tf.cast(y_true, tf.float32)) * y_pred, tf.float32))
    return tn / (tn + fp + tf.keras.backend.epsilon())

model = get_model()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

lesions_train, lesions_test, rtdoses_train, rtdoses_test, clinic_data_train, clinic_data_test, labels_train, labels_test = train_test_split(
    all_lesions, all_rtdoses, all_clinic_data, all_labels, test_size=.2
)

model.fit( [lesions_train, rtdoses_train, clinic_data_train], labels_train, epochs=50, batch_size=64, validation_split=.2 )
model.save("model.keras")

#model = keras.models.load_model("model.h5")

y_predict = model.predict([lesions_test, rtdoses_test, clinic_data_test])
y_predict = [1 if y > .5 else 0 for y in y_predict]

from sklearn.metrics import confusion_matrix
C = confusion_matrix(labels_test, y_predict)

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
