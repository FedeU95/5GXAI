import pandas as pd
import vote
import shap
import joblib
import numpy as np
import random

# Optional: pretty printing
from tabulate import tabulate

# Load model
model_path = 'xgb_model_5GNIDD.pkl'
m = joblib.load(model_path)

# Load TP samples
tp_csv_path = 'true_positives_balanced_5GNIDD.csv'
tp_with_multiclass_mapping = pd.read_csv(tp_csv_path)
print(f"True positives loaded from {tp_csv_path}")

# Prepare feature matrix and label column
X_tp = tp_with_multiclass_mapping.drop(columns=['Binary_Label', 'Multiclass_Label'])
labels = tp_with_multiclass_mapping['Multiclass_Label']
feature_names = X_tp.columns

# SHAP Explainer
shap_explainer = shap.TreeExplainer(m)

# Vote-XAI Explainer
vote_explainer = vote.Ensemble.from_xgboost(m)

# Get one random TP sample per class
samples_per_class = tp_with_multiclass_mapping.groupby('Multiclass_Label').apply(lambda df: df.sample(1, random_state=42)).reset_index(drop=True)

# Process each sample
for idx, row in samples_per_class.iterrows():
    class_label = row['Multiclass_Label']
    x_sample = row.drop(['Binary_Label', 'Multiclass_Label'])
    x_sample_df = x_sample.to_frame().T
    x_sample_array = x_sample_df.to_numpy()

    print("="*60)
    print(f"🛡️  Explaining Sample from Class: **{class_label}**\n")

    # -------- SHAP --------
    shap_values = shap_explainer.shap_values(x_sample_array)[0]
    shap_contribs = list(zip(feature_names, shap_values, x_sample_array[0]))

    # Sort by contribution value
    shap_contribs_sorted = sorted(shap_contribs, key=lambda x: (-x[1] if x[1] > 0 else float('inf'), abs(x[1])))

    print("🔍 SHAP Feature Contributions (positive first):")
    shap_table = [(feat, f"{val:.4f}", f"{feat_val:.4f}") for feat, val, feat_val in shap_contribs_sorted if val != 0]
    print(tabulate(shap_table, headers=["Feature", "SHAP Value", "Feature Value"], tablefmt="fancy_grid"))

    # -------- Vote-XAI --------
    vote_explanation = vote_explainer.explain_minimal(x_sample_array[0])
    vote_explanation_named = [(feature_names[i], x_sample_array[0][i]) for i in vote_explanation]

    print("\n🗳️  Vote-XAI Minimal Feature Subset:")
    vote_table = [(feat, f"{val:.4f}") for feat, val in vote_explanation_named]
    print(tabulate(vote_table, headers=["Feature", "Feature Value"], tablefmt="fancy_grid"))
    print("\n")
