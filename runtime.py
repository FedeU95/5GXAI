import pandas as pd
import vote
import shap
import joblib
import numpy as np
import time
import matplotlib.pyplot as plt

#STEP 6: Runtime Performance Comparison

model_path = 'xgb_model_PFCP.pkl'
#model_path = 'xgb_model_MSA.pkl'

m = joblib.load(model_path)


tp_csv_path = 'true_positives_balanced_large_PFCP.csv'
#tp_csv_path = 'true_positives_balanced_large_MSA.csv'

tp_with_multiclass_mapping = pd.read_csv(tp_csv_path)
print(f"True positives loaded from {tp_csv_path}")
X_tp_subset = tp_with_multiclass_mapping.drop(columns=['Binary_Label', 'Multiclass_Label'])

X_tp_subset = tp_with_multiclass_mapping.drop(columns=['Binary_Label', 'Multiclass_Label'])


# Vote-XAI
print("Starting Vote-XAI explanation...")

vote_explainer = vote.Ensemble.from_xgboost(m)

vote_times = []

for idx, row in X_tp_subset.iterrows():
    xvec = row.to_numpy()
    start_time = time.time()
    _ = vote_explainer.explain_minimal(xvec)
    end_time = time.time()
    vote_times.append(end_time - start_time)

# SHAP
print("Starting SHAP explanation...")
shap_explainer = shap.TreeExplainer(m)
shap_critical_features = []
shap_times = []

for idx, row in X_tp_subset.iterrows():
    xvec = row.to_numpy().reshape(1, -1)
    start_time = time.time()
    shap_values = shap_explainer.shap_values(xvec)[0]
    end_time = time.time()
    shap_times.append(end_time - start_time)
    critical_features = [feature for feature, shap_value in zip(X_tp_subset.columns, shap_values) if shap_value > 0]
    shap_critical_features.append(critical_features)

# Boxplot
plt.figure(figsize=(10, 6))  
plt.boxplot([shap_times, vote_times], labels=['SHAP', 'Vote-XAI'])

# Bigger fonts
plt.ylabel('Explanation Time (seconds)', fontsize=16)
plt.title('Runtime Performance PFCP', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.show()

#STATS

attack_labels = tp_with_multiclass_mapping['Multiclass_Label'].values


shap_feature_counts = [len(f) for f in shap_critical_features]
shap_min_idx = np.argmin(shap_feature_counts)

vote_feature_counts = [len(vote_explainer.explain_minimal(row.to_numpy())) for _, row in X_tp_subset.iterrows()]
vote_min_idx = np.argmin(vote_feature_counts)
import numpy as np

print("SHAP mean:", np.mean(shap_times))
print("Vote-XAI mean:", np.mean(vote_times))


print("\nSHAP Explanation Stats:")
print(f"  Min features: {min(shap_feature_counts)} (Attack class: {attack_labels[shap_min_idx]})")
print(f"  Max features: {max(shap_feature_counts)}")
print(f"  Mean features: {np.mean(shap_feature_counts):.2f}")

print("\nVote-XAI Explanation Stats:")
print(f"  Min features: {min(vote_feature_counts)} (Attack class: {attack_labels[vote_min_idx]})")
print(f"  Max features: {max(vote_feature_counts)}")
print(f"  Mean features: {np.mean(vote_feature_counts):.2f}")
