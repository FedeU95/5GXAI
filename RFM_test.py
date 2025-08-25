"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib

# STEP 1: Data Preparation 

data = pd.read_csv('5GNIDD.csv', low_memory=False)  # This is Combined.csv in the og 5GNIDD repo

# drop first unnamed/index column if present
data.drop(data.columns[0], axis=1, inplace=True)
data = data.drop_duplicates()

# Features and targets
X = data.iloc[:, :-2]  
y_binary = data.iloc[:, -2]  
y_multiclass = data.iloc[:, -1]  

# Encode targets
label_encoder_binary = LabelEncoder()
y_binary_encoded = label_encoder_binary.fit_transform(y_binary)

label_encoder_multiclass = LabelEncoder()
y_multiclass_encoded = label_encoder_multiclass.fit_transform(y_multiclass)

# Ensure malicious label is mapped to 1 (for your awareness)

try:
    malicious_label = label_encoder_binary.inverse_transform([1])
    print(f"Malicious label mapped to 1: {malicious_label}")
except Exception as e:
    print("Could not confirm label mapping:", e)

# STEP 2: Model training and testing

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    X, y_binary_encoded, range(len(y_multiclass_encoded)), test_size=0.2, random_state=42, stratify=y_binary_encoded
)

y_multiclass_test = y_multiclass_encoded[indices_test]

# Align columns (defensive)
X_test = X_test[X_train.columns]

# Random Forest parameters (tuned for a starting point)
rf_params = {
    "n_estimators": 400,
    "max_depth": None,           # let trees grow fully; adjust if overfitting
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": "balanced",  # handle class imbalance akin to scale_pos_weight
    "n_jobs": -1,
    "random_state": 42
}

m = RandomForestClassifier(**rf_params)
m.fit(X_train, y_train)

# Predictions & metrics
y_pred = m.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("TN: %s\t FP: %s\nFN: %s\t\t TP: %s" % (tn, fp, fn, tp))
print()
print("Accuracy:  %.4f" % (float(tp + tn) / len(y_test)))
print("Precision: %.4f" % (tp / float(tp + fp) if (tp + fp) > 0 else 0.0))
print("Recall:    %.4f" % (tp / float(tp + fn) if (tp + fn) > 0 else 0.0))

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib

# STEP 1: Data Preparation 

data = pd.read_csv('MSA.csv', low_memory=False)  # This is msa_nas.csv in the original repo, but with binary label added 

data.drop(data.columns[0], axis=1, inplace=True)
data = data.drop_duplicates()

# Multiclass label mapping
label_map = {
    0: 'Benign',
    1: 'Energy Depletion attack',
    2: 'NAS counter Desynch attack',
    3: 'X2 signalling flood',
    4: 'Paging channel hijacking attack',
    5: 'Bidding down with AttachReject',
    6: 'Incarceration with rrcReject and rrcRelease',
    7: 'Panic Attack',
    8: 'Stealthy Kickoff Attack',
    9: 'Authentication relay attack',
    10: 'Location tracking via measurement reports',
    11: 'Capability Hijacking',
    12: 'Lullaby attack using rrcReestablishRequest',
    13: 'Mobile Network Mapping (MNmap)',
    14: 'Lullaby attack with rrcResume',
    15: 'IMSI catching',
    16: 'Incarceration with rrcReestablishReject',
    17: 'Handover hijacking',
    18: 'RRC replay attack',
    19: 'Lullaby attack with rrcReconfiguration',
    20: 'Bidding down with ServiceReject',
    21: 'Bidding down with TAUReject'
}

data['Multiclass_Label'] = data['Multiclass_Label'].map(label_map)

# Feature & label separation
X = data.iloc[:, :-2]  
y_binary = data.iloc[:, -2]  
y_multiclass = data.iloc[:, -1]  

# STEP 2: Train/test split
X_train, X_test, y_train, y_test, y_multiclass_train, y_multiclass_test = train_test_split(
    X, y_binary, y_multiclass, test_size=0.2, random_state=42, stratify=y_binary
)

X_test = X_test[X_train.columns]

# Print dataset stats
print("==== TOTAL SAMPLE DISTRIBUTION ====")
print("Binary Label (Full Set):")
print(y_binary.value_counts().rename({0: 'Benign', 1: 'Malicious'}))
print("\nMulticlass Label (Full Set):")
print(y_multiclass.value_counts().sort_index())

print("\n==== TEST SET DISTRIBUTION ====")
print("Binary Label (Test Set):")
print(y_test.value_counts().rename({0: 'Benign', 1: 'Malicious'}))
print("\nMulticlass Label (Test Set):")
print(y_multiclass_test.value_counts().sort_index())

# STEP 3: Model training
rf_params = {
    "n_estimators": 400,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": "balanced",  # Handles imbalance
    "n_jobs": -1,
    "random_state": 42
}

m = RandomForestClassifier(**rf_params)
m.fit(X_train, y_train)

# STEP 4: Evaluation
y_pred = m.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("TN: %s\t FP: %s\nFN: %s\t\t TP: %s" % (tn, fp, fn, tp))
print()
print("Accuracy:  %.4f" % ((tp + tn) / len(y_test)))
print("Precision: %.4f" % (tp / (tp + fp) if (tp + fp) > 0 else 0.0))
print("Recall:    %.4f" % (tp / (tp + fn) if (tp + fn) > 0 else 0.0))
