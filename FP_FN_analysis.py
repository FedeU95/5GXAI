import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import joblib


data = pd.read_csv('5GNIDD.csv', low_memory=False)  
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

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    X, y_binary_encoded, range(len(y_multiclass_encoded)), test_size=0.2, random_state=42
)


y_multiclass_test = y_multiclass_encoded[indices_test]

X_test = X_test[X_train.columns]

model_path = 'xgb_model_5GNIDD.pkl'
m = joblib.load(model_path)
y_pred = m.predict(X_test)

# ==== False Positives ====
fp_indices = np.where((y_pred == 1) & (y_test == 0))[0]
fp_classes = y_multiclass_test[fp_indices]

X_fp = X_test.iloc[fp_indices].copy()
X_fp['Binary_Label'] = 0  # Actually benign
X_fp['Multiclass_Label'] = label_encoder_multiclass.inverse_transform(fp_classes)

fp_csv_path = 'false_positives_all_5GNIDD.csv'
X_fp.to_csv(fp_csv_path, index=False)
print(f"All {len(X_fp)} false positives saved to {fp_csv_path}")

print("FP class distribution:")
print(X_fp['Multiclass_Label'].value_counts())


# ==== False Negatives ====
fn_indices = np.where((y_pred == 0) & (y_test == 1))[0]
fn_classes = y_multiclass_test[fn_indices]

X_fn = X_test.iloc[fn_indices].copy()
X_fn['Binary_Label'] = 1  # Actually malicious
X_fn['Multiclass_Label'] = label_encoder_multiclass.inverse_transform(fn_classes)

fn_csv_path = 'false_negatives_all_5GNIDD.csv'
X_fn.to_csv(fn_csv_path, index=False)
print(f"All {len(X_fn)} false negatives saved to {fn_csv_path}")

print("FN class distribution:")
print(X_fn['Multiclass_Label'].value_counts())

from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("TN: %s\t FP: %s\nFN: %s\t\t TP: %s" % (tn, fp, fn, tp))
print()
print("Accuracy:  %.4f" % (float(tp + tn) / len(y_test)))
print("Precision: %.4f" % (tp / float(tp + fp)))
print("Recall:    %.4f" % (tp / float(tp + fn)))