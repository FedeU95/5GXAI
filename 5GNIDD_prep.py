import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import joblib


# STEP 1: Data Preparation 

data = pd.read_csv('5GNIDD.csv', low_memory=False)  # This is Combined.csv in the og 5GNIDD repo

data.drop(data.columns[0], axis=1, inplace=True)
data = data.drop_duplicates()


X = data.iloc[:, :-2]  
y_binary = data.iloc[:, -2]  
y_multiclass = data.iloc[:, -1]  

label_encoder_binary = LabelEncoder()
y_binary_encoded = label_encoder_binary.fit_transform(y_binary)

label_encoder_multiclass = LabelEncoder()
y_multiclass_encoded = label_encoder_multiclass.fit_transform(y_multiclass)

# Ensure malicious label is mapped to 1
malicious_label = label_encoder_binary.inverse_transform([1])
print(f"Malicious label mapped to 1: {malicious_label}")


# STEP 2: Model training and testing

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    X, y_binary_encoded, range(len(y_multiclass_encoded)), test_size=0.2, random_state=42
)


y_multiclass_test = y_multiclass_encoded[indices_test]

X_test = X_test[X_train.columns]

"""
params = {
    'scale_pos_weight': 2.2,
    'n_estimators': 100,
    'max_depth': 10,
    'learning_rate': 0.35
}

m = XGBClassifier(**params)
m.fit(X_train, y_train)

'''
# Save the model to avoid retraining
model_path = 'xgb_model_5GNIDD.pkl'
joblib.dump(m, model_path)
print(f"Model saved to {model_path}")

'''
y_pred = m.predict(X_test)
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("TN: %s\t FP: %s\nFN: %s\t\t TP: %s" % (tn, fp, fn, tp))
print()
print("Accuracy:  %.4f" % (float(tp + tn) / len(y_test)))
print("Precision: %.4f" % (tp / float(tp + fp)))
print("Recall:    %.4f" % (tp / float(tp + fn)))
"""


m = 'xgb_model_MSA.pkl'
y_pred = m.predict(X_test)

from sklearn.utils import resample

# STEP 3: Selection of True Positives for explanation
tp_indices = np.where((y_pred == 1) & (y_test == 1))[0]
tp_classes = y_multiclass_test[tp_indices]

tp_data = pd.DataFrame({
    'Index': tp_indices,
    'Class': tp_classes
})

print("Original TP class distribution:")
print(tp_data['Class'].value_counts())

# Stratified sampling setup
max_tp_samples = 100
tp_class_counts = tp_data['Class'].value_counts()
num_tp_classes = len(tp_class_counts)
samples_per_class = max_tp_samples // num_tp_classes

# Perform stratified sampling
selected_tp_indices = []
for cls, count in tp_class_counts.items():
    cls_indices = tp_data[tp_data['Class'] == cls]['Index']
    sampled_indices = resample(
        cls_indices,
        replace=(count < samples_per_class),
        n_samples=min(samples_per_class, count),
        random_state=42
    )
    selected_tp_indices.extend(sampled_indices)

# Subset data
X_tp_subset = X_test.iloc[selected_tp_indices]
y_multiclass_tp_subset = y_multiclass_test[selected_tp_indices]

# Add labels for output
X_tp_subset['Binary_Label'] = 1
X_tp_subset['Multiclass_Label'] = label_encoder_multiclass.inverse_transform(y_multiclass_tp_subset)

tp_with_multiclass_mapping = X_tp_subset.reset_index(drop=True)

# Save to CSV
tp_csv_path = 'true_positives_balanced_5GNIDD.csv'
tp_with_multiclass_mapping.to_csv(tp_csv_path, index=False)
print(f"True positives saved to {tp_csv_path}")

print("Balanced TP class distribution:")
print(tp_with_multiclass_mapping['Multiclass_Label'].value_counts())

#EXTRA: for runtime test - selection of larger TP subset

tp_indices_all = np.where((y_pred == 1) & (y_test == 1))[0]
tp_classes_all = y_multiclass_test[tp_indices_all]

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
y_multiclass_tp_subset_large = y_multiclass_test[selected_tp_indices_large]

# Final TP dataframe
X_tp_subset_large['Binary_Label'] = 1
X_tp_subset_large['Multiclass_Label'] = label_encoder_multiclass.inverse_transform(y_multiclass_tp_subset_large)

tp_with_multiclass_mapping_large = X_tp_subset_large.reset_index(drop=True)

tp_large_csv_path = 'true_positives_balanced_large_5GNIDD.csv'
tp_with_multiclass_mapping_large.to_csv(tp_large_csv_path, index=False)
print(f"Larger true positives saved to {tp_large_csv_path}")

print("Balanced class distribution (large TP set):")
print(tp_with_multiclass_mapping_large['Multiclass_Label'].value_counts())

"""
# Converting the model to json (needed to compute all minimal using vote-xai)
import vote
vote_explainer = vote.Ensemble.from_xgboost(m)

with open('5GNIDD.json', 'w') as f:
    f.write(vote_explainer.serialize())
"""

