import re
import pandas as pd
import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.utils.validation import check_is_fitted

# ------------------
# Load model & data
# ------------------
model_path = "xgb_model_PFCP.pkl"
m = joblib.load(model_path)

tp_csv_path = "true_positives_balanced_PFCP.csv"
tp_with_multiclass_mapping = pd.read_csv(tp_csv_path)

# Keep multiclass labels for grouping
multiclass_labels = tp_with_multiclass_mapping["Multiclass_Label"]

# Keep only columns that the model was actually trained on
# (protects against extra columns in the CSV)
model_cols = None
if hasattr(m, "feature_names_in_"):
    model_cols = list(m.feature_names_in_)
elif hasattr(m, "get_booster") and m.get_booster().feature_names is not None:
    model_cols = list(m.get_booster().feature_names)

if model_cols is None:
    # Fallback: drop obvious label/extra columns if model doesn't expose names
    drop_these = {"Binary_Label", "Multiclass_Label", "Attack_Description"}
    X_tp_subset = tp_with_multiclass_mapping.drop(columns=[c for c in drop_these if c in tp_with_multiclass_mapping.columns])
else:
    X_tp_subset = tp_with_multiclass_mapping.loc[:, model_cols].copy()

# ----------------------------
# Build base-name column map
# ----------------------------
# One-hot columns look like "Dst IP _172.21.0.120" or "Flow ID_172.21.0.107-...".
# We want the original dataset feature (before the first underscore).
def base_feature_name(col: str) -> str:
    # split at the first underscore; strip spaces
    parts = col.split("_", 1)
    return parts[0].strip()

col_to_base = {c: base_feature_name(c) for c in X_tp_subset.columns}
base_features = sorted(set(col_to_base.values()))

# ----------------------------
# SHAP per class (aggregated)
# ----------------------------
shap_explainer = shap.TreeExplainer(m)
shap_values_per_class = {}

for attack_class in multiclass_labels.unique():
    class_mask = (multiclass_labels == attack_class)
    idxs = np.where(class_mask)[0]

    # Collect per-sample SHAP values, aggregated to base features
    agg_rows = []
    for idx in idxs:
        xvec = X_tp_subset.iloc[idx:idx+1]  # 1 x n
        shap_vals = shap_explainer.shap_values(xvec)[0]   # (n_features,)

        # keep only positive contributions (like your code)
        shap_vals = np.where(shap_vals > 0, shap_vals, 0.0)

        # turn into Series with original encoded columns
        s = pd.Series(shap_vals, index=X_tp_subset.columns)

        # aggregate by base feature (sum positive contributions across its one-hot columns)
        agg = s.groupby(pd.Series(col_to_base)).sum()

        # ensure every base feature is present
        agg = agg.reindex(base_features, fill_value=0.0)
        agg_rows.append(agg)

    # stack rows => DataFrame [n_samples_in_class x n_base_features]
    agg_df = pd.DataFrame(agg_rows, columns=base_features)

    # stats across samples for this attack class
    stats = pd.DataFrame(index=base_features)
    stats["Min"] = agg_df.min(axis=0)
    stats["Max"] = agg_df.max(axis=0)
    stats["Mean"] = agg_df.mean(axis=0)

    shap_values_per_class[attack_class] = stats

# ----------------------------
# Flatten to CSV (Mean/Min/Max)
# ----------------------------
flat = pd.DataFrame(index=base_features)
for attack_class, stats in shap_values_per_class.items():
    flat[f"{attack_class}_Mean"] = stats["Mean"]
    flat[f"{attack_class}_Min"]  = stats["Min"]
    flat[f"{attack_class}_Max"]  = stats["Max"]

flat = flat.reset_index().rename(columns={"index": "Feature"})

out_csv = "shap_min_max_mean_per_feature_PFCP.csv"
flat.to_csv(out_csv, index=False)
print(f"\nSaved flat SHAP stats (aggregated by base feature) to: {out_csv}")

# ----------------------------
# Pretty plot: Top-N per class
# ----------------------------
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 28,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18
})

TOP_N = 5
use_log_scale = False

shap_df = flat.copy()
attack_classes = sorted(set(c.split("_")[0] for c in shap_df.columns if c.endswith("_Mean")))

print("\nTop 5 Features by SHAP Mean for Each Attack Class (with Min and Max):\n")
for attack_class in attack_classes:
    mean_col = f"{attack_class}_Mean"
    min_col  = f"{attack_class}_Min"
    max_col  = f"{attack_class}_Max"
    if not all(col in shap_df.columns for col in [mean_col, min_col, max_col]): 
        continue

    top = shap_df[["Feature", mean_col, min_col, max_col]].nlargest(TOP_N, mean_col)
    top = top.rename(columns={mean_col: "Mean", min_col: "Min", max_col: "Max"})

    print(f"Attack Class: {attack_class}")
    print(top.to_string(index=False))
    print("-" * 60)

    features = top["Feature"].to_numpy()
    means    = top["Mean"].to_numpy()
    mins     = top["Min"].to_numpy()
    maxs     = top["Max"].to_numpy()

    x = np.arange(len(features))
    fig, ax = plt.subplots(figsize=(14, 9))

    bars = ax.bar(x, means, zorder=2)
    ax.scatter(x, mins, marker="o", s=180, c="black", linewidths=2.5, zorder=4, label="Min")
    ax.scatter(x, maxs, marker="x", s=220, c="black", linewidths=3.5, zorder=4, label="Max")

    if use_log_scale:
        ax.set_yscale("log")

    ax.set_ylabel("SHAP Value")
    ax.set_title(f"Top {TOP_N} SHAP Features — {attack_class}")
    ax.set_xticks(x, features, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=1)

    legend_handles = [
        bars[0],
        Line2D([0], [0], marker="o", linestyle="None", markersize=12,
               markerfacecolor="black", markeredgecolor="black", label="Min"),
        Line2D([0], [0], marker="x", linestyle="None", markersize=12,
               color="black", markeredgewidth=3, label="Max"),
    ]
    ax.legend(legend_handles, ["Mean", "Min", "Max"], frameon=False)

    plt.tight_layout()
    plt.show()
