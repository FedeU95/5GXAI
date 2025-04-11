import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
#TO DO: MAP WITH THE CATEGORIC LABELS BEFORE ANYTHING

X = data.iloc[:, :-2]  
y_binary = data.iloc[:, -2]  
y_multiclass = data.iloc[:, -1]  

"""
#encoding not needed here 

label_encoder_binary = LabelEncoder()
y_binary_encoded = label_encoder_binary.fit_transform(y_binary)

label_encoder_multiclass = LabelEncoder()
y_multiclass_encoded = label_encoder_multiclass.fit_transform(y_multiclass)

# Ensure malicious label is mapped to 1
malicious_label = label_encoder_binary.inverse_transform([1])
print(f"Malicious label mapped to 1: {malicious_label}")
"""

# STEP 2: Model training and testing

X_train, X_test, y_train, y_test, y_multiclass_train, y_multiclass_test = train_test_split(
    X, y_binary, y_multiclass, test_size=0.2, random_state=42
)


X_test = X_test[X_train.columns]



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
"""

params = {
    'scale_pos_weight': 2.2,
    'n_estimators': 100,
    'max_depth': 10,
    'learning_rate': 0.35
}
FEATURE_START, FEATURE_END = 0, X.shape[1]
params['monotone_constraints'] = tuple([1] * (FEATURE_END - FEATURE_START))
m = XGBClassifier(**params)
m.fit(X_train, y_train)


# Save the model to avoid retraining
model_path = 'xgb_model_MSA.pkl'
joblib.dump(m, model_path)
print(f"Model saved to {model_path}")



y_pred = m.predict(X_test)
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("TN: %s\t FP: %s\nFN: %s\t\t TP: %s" % (tn, fp, fn, tp))
print()
print("Accuracy:  %.4f" % (float(tp + tn) / len(y_test)))
print("Precision: %.4f" % (tp / float(tp + fp)))
print("Recall:    %.4f" % (tp / float(tp + fn)))
"""

model_path = 'xgb_model_MSA.pkl'
m = joblib.load(model_path)
y_pred = m.predict(X_test)

from sklearn.utils import resample
"""
# STEP 3: Selection of True Positives for explanation

# FOR THIS PART I WILL SELECT A SUBSET OF THE 21 ATTACK CLASSES FOR THE SAKE OF SIMPLICITY, THERE ARE 9 MACRO CATEGORIES SO 1 SET OF X SAMPLES PER EACH CATEGORY 

tp_indices = np.where((y_pred == 1) & (y_test == 1))[0]
tp_classes = y_multiclass_test.iloc[tp_indices]

tp_data = pd.DataFrame({
    'Index': tp_indices,
    'Class': tp_classes
})

print("Original TP class distribution:")
print(tp_data['Class'].value_counts())

# === FINAL SELECTED ATTACK CLASSES ===
selected_attacks = [
    'NAS counter Desynch attack',
    'Location tracking via measurement reports',
    'RRC replay attack',
    'Energy Depletion attack',
    'Authentication relay attack',
    'Handover hijacking',
    'Lullaby attack using rrcReestablishRequest',
    'Bidding down with TAUReject',
    'Bidding down with AttachReject',
    'Incarceration with rrcReestablishReject'
]

samples_per_class = 8

# STEP 1: Filter true positives only
tp_indices = np.where((y_pred == 1) & (y_test == 1))[0]
tp_classes = y_multiclass_test.iloc[tp_indices]

tp_data = pd.DataFrame({
    'Index': tp_indices,
    'Class': tp_classes
})

# STEP 2: Sample 8 TPs per selected class
selected_tp_indices = []
for attack_name in selected_attacks:
    class_indices = tp_data[tp_data['Class'] == attack_name]['Index']
    if len(class_indices) == 0:
        print(f"Warning: No TP samples found for class: {attack_name}")
        continue
    sampled_indices = resample(
        class_indices,
        replace=(len(class_indices) < samples_per_class),
        n_samples=min(samples_per_class, len(class_indices)),
        random_state=42
    )
    selected_tp_indices.extend(sampled_indices)

# STEP 3: Subset data
X_tp_subset = X_test.iloc[selected_tp_indices]
y_multiclass_tp_subset = y_multiclass_test.iloc[selected_tp_indices]

# Add labels for output
X_tp_subset['Binary_Label'] = 1
X_tp_subset['Multiclass_Label'] = y_multiclass_tp_subset

# Save to CSV
tp_csv_path = 'true_positives_balanced_MSA.csv'
X_tp_subset.reset_index(drop=True).to_csv(tp_csv_path, index=False)

print(f"True positives saved to {tp_csv_path}")
print("\nBalanced TP class distribution:")
print(X_tp_subset['Multiclass_Label'].value_counts())

"""
"""
#EXTRA: for runtime test - selection of larger TP subset

tp_indices_all = np.where((y_pred == 1) & (y_test == 1))[0]
tp_classes_all = y_multiclass_test.iloc[tp_indices_all]

tp_data_all = pd.DataFrame({
    'Index': tp_indices_all,
    'Class': tp_classes_all
})

print("\nOriginal TP class distribution (for extended TP set):")
print(tp_data_all['Class'].value_counts())

# Define a larger sample size for TPs (1200)
max_tp_samples_large = 1200
tp_class_counts = tp_data_all['Class'].value_counts()
num_tp_classes = len(tp_class_counts)
samples_per_class_large = max_tp_samples_large // num_tp_classes

# Stratified sampling
selected_tp_indices_large = []
for attack_class, count in tp_class_counts.items():
    class_indices = tp_data_all[tp_data_all['Class'] == attack_class]
    selected_indices = resample(
        class_indices['Index'],
        replace=(count < samples_per_class_large),
        n_samples=min(samples_per_class_large, count),
        random_state=42
    )
    selected_tp_indices_large.extend(selected_indices)

X_tp_subset_large = X_test.iloc[selected_tp_indices_large]
y_multiclass_tp_subset_large = y_multiclass_test.iloc[selected_tp_indices_large]

# Final TP dataframe
X_tp_subset_large['Binary_Label'] = 1
X_tp_subset_large['Multiclass_Label'] = y_multiclass_tp_subset_large

tp_with_multiclass_mapping_large = X_tp_subset_large.reset_index(drop=True)

tp_large_csv_path = 'true_positives_balanced_large_MSA.csv'
tp_with_multiclass_mapping_large.to_csv(tp_large_csv_path, index=False)
print(f"Larger true positives saved to {tp_large_csv_path}")

print("Balanced class distribution (large TP set):")
tp_with_multiclass_mapping_large['Attack_Name'] = tp_with_multiclass_mapping_large['Multiclass_Label'].map(label_map)
print(tp_with_multiclass_mapping_large['Multiclass_Label'].value_counts())

"""
# Converting the model to json (needed to compute all minimal using vote-xai)
import vote
vote_explainer = vote.Ensemble.from_xgboost(m)

with open('MSA.json', 'w') as f:
    f.write(vote_explainer.serialize())



