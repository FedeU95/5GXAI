import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

#STEP 4 - 5: Explanations experiments (Vote-XAI)

# Load data
vote_xai_features = pd.read_csv("top_features_per_class_MSA_new.csv", index_col=0)  # see all_minimal.py
shap_stats = pd.read_csv("shap_min_max_mean_per_feature_MSA.csv") # see shap_study.py

# Restructure SHAP dataframe: set Feature as index
shap_stats.set_index("Feature", inplace=True)

# Extract class names
class_labels = sorted(set(col.split("_")[0] for col in shap_stats.columns if col.endswith("Mean")))

for class_label in class_labels:
    print(f"\n===== Class: {class_label} =====")

    # Column names
    shap_col = f"{class_label}_Mean"

    # Get top 5 Vote-XAI features (by occurrence)
    top_vote_features = vote_xai_features[class_label].sort_values(ascending=False).head(5)

    print("\n-- Top 5 Vote-XAI Features --")
    vote_table = pd.DataFrame({
        "Vote-XAI Occurrence (%)": (top_vote_features * 100).round(1),
        "SHAP Mean Value": shap_stats.loc[top_vote_features.index, shap_col].round(4)
    })
    print(vote_table)

    # Get top 5 SHAP features
    top_shap_features = shap_stats[shap_col].sort_values(ascending=False).head(5)

    print("\n-- Top 5 SHAP Features --")
    shap_table = pd.DataFrame({
        "SHAP Mean Value": top_shap_features.round(4),
        "Vote-XAI Occurrence (%)": (vote_xai_features[class_label].loc[top_shap_features.index] * 100).round(1)
    })
    print(shap_table)

for class_label in vote_xai_features.columns:
    # Get top features based on occurrence (>90%)
    top_features = vote_xai_features[class_label][vote_xai_features[class_label] > 0.9].index.tolist()
    if not top_features:
        continue

    # Limit to top 10 based on Vote-XAI occurrence
    top_features = sorted(
        top_features,
        key=lambda f: vote_xai_features[class_label][f],
        reverse=True
    )[:10]

    # Get SHAP mean values for top features
    shap_col = f"{class_label}_Mean"
    shap_means = shap_stats.loc[top_features, shap_col]
    occurrences = vote_xai_features[class_label].loc[top_features] * 100  # Convert to percentage

    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()

    width = 0.4
    x = np.arange(len(top_features))

    # Plot raw values
    bars1 = ax1.bar(x - width/2, occurrences, width, color="green", label="Vote-XAI")
    bars2 = ax2.bar(x + width/2, shap_means, width, color="blue", label="SHAP")

    # Set both axes to log scale
    ax1.set_yscale("log")
    ax2.set_yscale("log")

    # Match y-axis limits and ticks (based on global range)
    ax1.set_ylim(0.01, 100)
    ax2.set_ylim(0.01, 100)

    ticks = [0.01, 0.1, 1, 10, 100]
    ax1.set_yticks(ticks)
    ax2.set_yticks(ticks)

    # Axis labels
    ax1.set_ylabel("Vote-XAI Occurrence (%)", fontsize=22, color="green")
    ax2.set_ylabel("SHAP Mean Value", fontsize=22, color="blue")

    # X-axis
    ax1.set_xticks(x)
    ax1.set_xticklabels(top_features, fontsize=16)

    # Ticks and styling
    ax1.tick_params(axis='y', labelsize=20, colors="green")
    ax2.tick_params(axis='y', labelsize=20, colors="blue")
    ax1.tick_params(axis='x', labelsize=14, labelrotation=45)

    plt.title(f"Vote-XAI Occurrence and Mean SHAP Values ({class_label})", fontsize=24)
    green_patch = mpatches.Patch(color='green', label='Vote-XAI (%)')
    blue_patch = mpatches.Patch(color='blue', label='SHAP')
    plt.legend(handles=[green_patch, blue_patch], fontsize=18, loc="upper right")

    plt.tight_layout()
    plt.show()