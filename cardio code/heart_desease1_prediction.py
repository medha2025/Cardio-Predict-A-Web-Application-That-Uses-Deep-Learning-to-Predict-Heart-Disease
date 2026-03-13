# ===============================
# HEART DISEASE PREDICTION USING DEEP LEARNING
# (Enhanced with Precision, Recall, ROC-AUC visualization and individual prediction)
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ===============================
# 1. LOAD & PREPROCESS DATA
# ===============================

data = pd.read_csv('heart_1000.csv')  # Changed file name here
print("Original Shape:", data.shape)
print("Columns:", data.columns)

# Replace missing values
data.replace('?', np.nan, inplace=True)
data = data.dropna()
print("After Removing Missing Values:", data.shape)

# Convert target to binary
data['target'] = (data['target'] > 0).astype(int)

# Split features and target
X = data.drop('target', axis=1)
y = data['target']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ===============================
# 2. BUILD THE MODEL
# ===============================

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ===============================
# 3. TRAIN MODEL WITH EARLY STOPPING
# ===============================

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ===============================
# 4. PLOT ACCURACY, LOSS, PRECISION & RECALL
# ===============================

# Predict probabilities on test set
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Compute precision-recall curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall_vals, precision_vals)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(18,5))

# Accuracy & Loss
plt.subplot(1,3,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Accuracy & Loss Over Epochs")
plt.xlabel("Epochs")
plt.legend()

# Precision & Recall
plt.subplot(1,3,2)
plt.plot(precision_vals, label='Precision')
plt.plot(recall_vals, label='Recall')
plt.title(f"Precision-Recall Curve (AUC={pr_auc:.3f})")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()

# ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.subplot(1,3,3)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC={roc_auc:.3f})')
plt.plot([0,1], [0,1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.tight_layout()
plt.show()

# ===============================
# 5. MODEL EVALUATION
# ===============================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")
print(f"Accuracy:   {accuracy:.3f}")
print(f"Precision:  {precision:.3f}")
print(f"Recall:     {recall:.3f}")
print(f"F1 Score:   {f1:.3f}")
print(f"ROC AUC:    {roc_auc:.3f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ===============================
# 6. INDIVIDUAL PREDICTION
# ===============================

def predict_person(features):
    """
    Input: features = list or array of 13 feature values in order
    Output: 0 = No Heart Disease, 1 = Heart Disease
    """
    features_scaled = scaler.transform([features])
    prediction_prob = model.predict(features_scaled)[0][0]
    prediction = int(prediction_prob > 0.5)
    return prediction, prediction_prob

# Example usage:
# Replace the list below with actual patient data in the same order as the dataset columns
example_patient = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
pred_class, pred_prob = predict_person(example_patient)
print(f"\nIndividual Prediction: {'Heart Disease (1)' if pred_class==1 else 'No Heart Disease (0)'} with probability {pred_prob:.3f}")
