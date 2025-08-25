import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
import joblib


TRAIN_PATH = "PFCP/Training_net.csv"
TEST_PATH  = "PFCP/Testing_net.csv"
LABEL_COL  = "Label"          # original (multi-class) label column name
TEST_SIZE  = 0.20             # 80:20 split
RANDOM_STATE = 42

# Saving options
SAVE_MODEL       = True
MODEL_PATH       = "xgb_model_PFCP.pkl"

# XGBoost params (scale_pos_weight computed from the TRAIN split below)
XGB_PARAMS = dict(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.35,
    scale_pos_weight= 2.2,
)

# ========================
# 1) LOAD + MERGE + CLEAN
# ========================
train_raw = pd.read_csv(TRAIN_PATH, low_memory=False)
test_raw  = pd.read_csv(TEST_PATH,  low_memory=False)

# Drop only the first column (match your original behavior)
train = train_raw.drop(train_raw.columns[0], axis=1).copy()
test  = test_raw.drop(test_raw.columns[0], axis=1).copy()

from sklearn.preprocessing import LabelEncoder

# Merge and drop duplicates
df = pd.concat([train, test], axis=0, ignore_index=True)
df = df.drop_duplicates()

# Keep the multiclass labels around
label_encoder_multiclass = LabelEncoder()
y_multiclass_full = label_encoder_multiclass.fit_transform(df[LABEL_COL])

# Sanity check
if LABEL_COL not in df.columns:
    raise ValueError(f"'{LABEL_COL}' must exist in the merged data.")

# ========================
# 2) BUILD BINARY TARGET
# ========================
df["y_binary"] = (df[LABEL_COL] != "Normal").astype(int)


# ========================
# 3) FEATURE/TARGET SPLIT
# ========================
X = df.drop(columns=[LABEL_COL, "y_binary"])
y = df["y_binary"].values

# One-hot encode categoricals (fit on ALL data first, then split)
X_enc = pd.get_dummies(X, drop_first=False)

# ========================
# 4) STRATIFIED SPLIT (80:20)
# ========================
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_enc, y, np.arange(len(df)), stratify=y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Multiclass labels aligned to the test indices from the split
y_multiclass_test = y_multiclass_full[idx_test]

model = XGBClassifier(**XGB_PARAMS)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("TN: %s\t FP: %s" % (tn, fp))
print("FN: %s\t\t TP: %s" % (fn, tp))
print()
accuracy = (tp + tn) / len(y_test)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = f1_score(y_test, y_pred)

print("Accuracy:  %.4f" % accuracy)
print("Precision: %.4f" % precision)
print("Recall:    %.4f" % recall)
print("F1-Score:  %.4f" % f1)
# Class distribution check
def pct(x): 
    return f"{100.0 * x:.2f}%"
full_pos = y.mean()
train_pos = y_train.mean()
test_pos = y_test.mean()
print("\nClass distribution (positive=1):")
print(f"Full:  {pct(full_pos)}   Train: {pct(train_pos)}   Test: {pct(test_pos)}")

if SAVE_MODEL:
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")



from sklearn.utils import resample

# ---- TP selection & stratified sampling over MULTICLASS on the TEST SET ----
tp_mask = (y_pred == 1) & (y_test == 1)
tp_indices = np.where(tp_mask)[0]
tp_classes = y_multiclass_test[tp_indices]

tp_data = pd.DataFrame({
    "Index": tp_indices,
    "ClassIdx": tp_classes
})
tp_data["Class"] = label_encoder_multiclass.inverse_transform(tp_data["ClassIdx"])

print("Original TP class distribution:")
print(tp_data["Class"].value_counts())

# Small balanced TP subset
max_tp_samples = 100
tp_class_counts = tp_data["Class"].value_counts()
num_tp_classes = len(tp_class_counts)
samples_per_class = max(1, max_tp_samples // num_tp_classes)

selected_tp_indices = []
for cls, count in tp_class_counts.items():
    cls_indices = tp_data.loc[tp_data["Class"] == cls, "Index"]
    n = min(samples_per_class, len(cls_indices))
    sampled_indices = resample(
        cls_indices,
        replace=(len(cls_indices) < n),
        n_samples=n,
        random_state=42
    )
    selected_tp_indices.extend(sampled_indices.tolist())

selected_tp_indices = sorted(set(selected_tp_indices))

# Subset encoded test features; attach labels
X_tp_subset = X_test.iloc[selected_tp_indices].copy()
y_multiclass_tp_subset = y_multiclass_test[selected_tp_indices]

X_tp_subset["Binary_Label"] = 1
X_tp_subset["Multiclass_Label"] = label_encoder_multiclass.inverse_transform(y_multiclass_tp_subset)

# Optional: map to full attack descriptions
label_mapping = {
    "Mal_Estab": "PFCP Session Establishment Flood",
    "Mal_Del":   "PFCP Session Deletion Flood",
    "Mal_Mod":   "PFCP Session Modification Flood (DROP)",
    "Mal_Mod2":  "PFCP Session Modification Flood (DUPL)",
    "Normal":    "Normal traffic flow",
}
X_tp_subset["Multiclass_Label"] = X_tp_subset["Multiclass_Label"].map(label_mapping)

tp_with_multiclass_mapping = X_tp_subset.reset_index(drop=True)

# Save to CSV
tp_csv_path = "true_positives_balanced_PFCP.csv"
tp_with_multiclass_mapping.to_csv(tp_csv_path, index=False)
print(f"True positives saved to {tp_csv_path}")

print("Balanced TP class distribution:")
print(tp_with_multiclass_mapping["Multiclass_Label"].value_counts())

# ---- Larger TP subset (e.g., for runtime tests) ----
tp_indices_all = np.where(tp_mask)[0]
tp_classes_all = y_multiclass_test[tp_indices_all]

tp_data_all = pd.DataFrame({
    "Index": tp_indices_all,
    "ClassIdx": tp_classes_all
})
tp_data_all["Class"] = label_encoder_multiclass.inverse_transform(tp_data_all["ClassIdx"])

print("\nOriginal TP class distribution (for extended TP set):")
print(tp_data_all["Class"].value_counts())

max_tp_samples_large = 1200
tp_class_counts_large = tp_data_all["Class"].value_counts()
num_tp_classes_large = len(tp_class_counts_large)
samples_per_class_large = max(1, max_tp_samples_large // num_tp_classes_large)

selected_tp_indices_large = []
for cls, count in tp_class_counts_large.items():
    cls_indices = tp_data_all.loc[tp_data_all["Class"] == cls, "Index"]
    n = min(samples_per_class_large, len(cls_indices))
    sampled = resample(
        cls_indices,
        replace=(len(cls_indices) < n),
        n_samples=n,
        random_state=42
    )
    selected_tp_indices_large.extend(sampled.tolist())

selected_tp_indices_large = sorted(set(selected_tp_indices_large))

X_tp_subset_large = X_test.iloc[selected_tp_indices_large].copy()
y_multiclass_tp_subset_large = y_multiclass_test[selected_tp_indices_large]

X_tp_subset_large["Binary_Label"] = 1
X_tp_subset_large["Multiclass_Label"] = label_encoder_multiclass.inverse_transform(y_multiclass_tp_subset_large)
X_tp_subset_large["Multiclass_Label"] = X_tp_subset_large["Multiclass_Label"].map(label_mapping)

tp_with_multiclass_mapping_large = X_tp_subset_large.reset_index(drop=True)

tp_large_csv_path = "true_positives_balanced_large_PFCP.csv"
tp_with_multiclass_mapping_large.to_csv(tp_large_csv_path, index=False)
print(f"Larger true positives saved to {tp_large_csv_path}")

print("Balanced class distribution (large TP set):")
print(tp_with_multiclass_mapping_large["Multiclass_Label"].value_counts())

try:
    import vote
    vote_explainer = vote.Ensemble.from_xgboost(model)
    with open("PFCP.json", "w") as f:
        f.write(vote_explainer.serialize())
    print("Saved PFCP.json")
except Exception as e:
    print(f"vote-xai export skipped: {e}")
