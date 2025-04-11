import pandas as pd
import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import shap
import joblib
import numpy as np

# STEP 4 - 5: SHAP Explanations
"""
model_path = 'xgb_model_5GNIDD.pkl'
#model_path = 'xgb_model_MSA.pkl'
m = joblib.load(model_path)

tp_csv_path = 'true_positives_balanced_5GNIDD.csv'
tp_with_multiclass_mapping = pd.read_csv(tp_csv_path)

X_tp_subset = tp_with_multiclass_mapping.drop(columns=['Binary_Label', 'Multiclass_Label'])
multiclass_labels = tp_with_multiclass_mapping['Multiclass_Label']

shap_values_per_class = {}

# SHAP 
shap_explainer = shap.TreeExplainer(m)

for attack_class in tp_with_multiclass_mapping['Multiclass_Label'].unique():
    class_indices = tp_with_multiclass_mapping['Multiclass_Label'] == attack_class
 
    shap_class_values = []
    
    for idx in class_indices[class_indices].index:
        xvec = X_tp_subset.iloc[idx].to_numpy().reshape(1, -1)
        shap_values = shap_explainer.shap_values(xvec)[0]
        
        # Consider only positive SHAP values
        shap_values = np.where(shap_values > 0, shap_values, 0)
        shap_class_values.append(shap_values)
    
    shap_class_df = pd.DataFrame(shap_class_values, columns=X_tp_subset.columns)
    
    # Compute stats
    shap_stats = pd.DataFrame(index=X_tp_subset.columns)
    shap_stats["Min"] = shap_class_df.min(axis=0)
    shap_stats["Max"] = shap_class_df.max(axis=0)
    shap_stats["Mean"] = shap_class_df.mean(axis=0)
    
    shap_values_per_class[attack_class] = shap_stats

# Reorder columns for each class: Mean, Min, Max

flat_stats = pd.DataFrame(index=X_tp_subset.columns)

for attack_class, stats in shap_values_per_class.items():
    flat_stats[f"{attack_class}_Mean"] = stats["Mean"]
    flat_stats[f"{attack_class}_Min"] = stats["Min"]
    flat_stats[f"{attack_class}_Max"] = stats["Max"]


flat_stats = flat_stats.reset_index().rename(columns={'index': 'Feature'})


output_file_flat = "shap_min_max_mean_per_feature_MSA.csv"
flat_stats.to_csv(output_file_flat, index=False)
print(f"\nSaved flat SHAP stats to: {output_file_flat}")

"""
# Post-explanations: look at the top 5 per class


csv_path = "shap_min_max_mean_per_feature_MSA.csv"
shap_df = pd.read_csv(csv_path)

attack_classes = sorted(set(col.split("_")[0] for col in shap_df.columns if col.endswith("_Mean")))

print("\nTop 5 Features by SHAP Mean for Each Attack Class (with Min and Max):\n")

for attack_class in attack_classes:
    mean_col = f"{attack_class}_Mean"
    min_col = f"{attack_class}_Min"
    max_col = f"{attack_class}_Max"
    
    if all(col in shap_df.columns for col in [mean_col, min_col, max_col]):
        top_features = shap_df[["Feature", mean_col, min_col, max_col]].nlargest(5, mean_col)
        top_features = top_features.rename(columns={
            mean_col: "Mean",
            min_col: "Min",
            max_col: "Max"
        })
        
        print(f"Attack Class: {attack_class}")
        print(top_features.to_string(index=False))
        print("-" * 60)


import matplotlib.pyplot as plt
import numpy as np

# Plot settings
TOP_N = 5  # Change to 10 if desired
use_log_scale = False  # Set to True to use log scale for SHAP values

for attack_class in attack_classes:
    mean_col = f"{attack_class}_Mean"
    min_col = f"{attack_class}_Min"
    max_col = f"{attack_class}_Max"
    
    if all(col in shap_df.columns for col in [mean_col, min_col, max_col]):
        top_features = shap_df[["Feature", mean_col, min_col, max_col]].nlargest(TOP_N, mean_col)

        features = top_features["Feature"]
        means = top_features[mean_col]
        mins = top_features[min_col]
        maxs = top_features[max_col]

        # Calculate error bars (asymmetrical)
        error_lower = means - mins
        error_upper = maxs - means
        error = [error_lower, error_upper]

        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.bar(features, means, yerr=error, capsize=6, color="skyblue", edgecolor="black")

        if use_log_scale:
            ax.set_yscale("log")

        ax.set_ylabel("SHAP Value", fontsize=20)
        ax.set_title(f"Top {TOP_N} SHAP Features — {attack_class}", fontsize=24)
        ax.tick_params(axis='x', labelsize = 14, labelrotation= 45)
        ax.tick_params(axis='y', labelsize=18)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()
