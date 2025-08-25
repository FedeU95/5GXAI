import pandas as pd
import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt

COHORT = "FN"  # set to "FP" or "FN"
csv_path = 'false_negatives_all_5GNIDD.csv'  # change to 'false_negatives_all_5GNIDD.csv' for FN
# ============================================

# Model
model_path = 'xgb_model_5GNIDD.pkl'  
m = joblib.load(model_path)

# Load cohort 
cohort_df = pd.read_csv(csv_path)

X_subset = cohort_df.drop(columns=['Binary_Label', 'Multiclass_Label'])
multiclass_labels = cohort_df['Multiclass_Label']

shap_values_per_class = {}

# SHAP
shap_explainer = shap.TreeExplainer(m)

for attack_class in cohort_df['Multiclass_Label'].unique():
    class_indices = cohort_df['Multiclass_Label'] == attack_class
 
    shap_class_values = []
    
    for idx in class_indices[class_indices].index:
        xvec = X_subset.iloc[idx].to_numpy().reshape(1, -1)
        shap_values = shap_explainer.shap_values(xvec)[0]
        shap_values = np.where(shap_values > 0, shap_values, 0)
        # =================================================

        shap_class_values.append(shap_values)
    
    shap_class_df = pd.DataFrame(shap_class_values, columns=X_subset.columns)
    
    # Compute stats
    shap_stats = pd.DataFrame(index=X_subset.columns)
    shap_stats["Min"] = shap_class_df.min(axis=0)
    shap_stats["Max"] = shap_class_df.max(axis=0)
    shap_stats["Mean"] = shap_class_df.mean(axis=0)
    
    shap_values_per_class[attack_class] = shap_stats

# Flatten (Mean, Min, Max per class)
flat_stats = pd.DataFrame(index=X_subset.columns)

for attack_class, stats in shap_values_per_class.items():
    flat_stats[f"{attack_class}_Mean"] = stats["Mean"]
    flat_stats[f"{attack_class}_Min"] = stats["Min"]
    flat_stats[f"{attack_class}_Max"] = stats["Max"]

flat_stats = flat_stats.reset_index().rename(columns={'index': 'Feature'})

# Output file reflects cohort
output_file_flat = f"shap_min_max_mean_{COHORT}_5GNIDD.csv"
flat_stats.to_csv(output_file_flat, index=False)
print(f"\nSaved flat SHAP stats to: {output_file_flat}")

# -------- Post-explanations: top-N per class--------

import numpy as np
import matplotlib.pyplot as plt

TOP_N = 5
use_log_scale = False

attack_classes = sorted(set(col.split("_")[0] for col in flat_stats.columns if col.endswith("_Mean")))

for attack_class in attack_classes:
    mean_col = f"{attack_class}_Mean"
    min_col  = f"{attack_class}_Min"
    max_col  = f"{attack_class}_Max"
    
    if all(col in flat_stats.columns for col in [mean_col, min_col, max_col]):
        # Top-N table for this class
        top_features = flat_stats[["Feature", mean_col, min_col, max_col]].nlargest(TOP_N, mean_col)
        top_features = top_features.rename(columns={mean_col:"Mean", min_col:"Min", max_col:"Max"})
        
        print(f"Class: {attack_class}  | Cohort: {COHORT}")
        print(top_features.to_string(index=False))
        print("-" * 60)

        # ---- PLOT with safe error bars ----
        features = top_features["Feature"].astype(str).tolist()
        means = pd.to_numeric(top_features["Mean"], errors="coerce").fillna(0)
        mins  = pd.to_numeric(top_features["Min"],  errors="coerce").fillna(0)
        maxs  = pd.to_numeric(top_features["Max"],  errors="coerce").fillna(0)

        # Guard against tiny negatives from floating-point noise
        error_lower = np.clip((means - mins).to_numpy(), 0, None)
        error_upper = np.clip((maxs - means).to_numpy(), 0, None)
        error = [error_lower, error_upper]

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.bar(features, means.to_numpy(), yerr=error, capsize=6, edgecolor="black")

        if use_log_scale:
            ax.set_yscale("log")

        ylabel = "SHAP (towards malicious)" if COHORT == "FP" else "SHAP (away from malicious)"
        ax.set_ylabel(ylabel, fontsize=20)
        ax.set_title(f"Top {TOP_N} SHAP Features — {attack_class} ({COHORT})", fontsize=24)
        ax.tick_params(axis='x', labelsize=14, labelrotation=45)
        ax.tick_params(axis='y', labelsize=18)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()
