import os
import json
import numpy as np
import pandas as pd

# The scipt run_vote compute all minimal using python
# The json files contain all minimal explanations per class computed using the vote-xai binary
# To compute all minimal: 
# in /src: ./vote_explain -m test-model.json test-samples.csv --all -o explanations.json

#json_dir = "./json"
json_dir = "./json/PFCP"
#json_dir = "./json/MSA_json"
#output_file = "top_features_per_class_MSA_new.csv"
output_file = "top_features_PFCP.csv"
csv_path = "true_positives_balanced_PFCP.csv"
#csv_path = 'true_positives_balanced_MSA.csv'
df = pd.read_csv(csv_path)

feature_names = df.drop(columns=['Binary_Label', 'Multiclass_Label']).columns.tolist()

# --------------------------
# MAP ONE-HOT -> BASE NAMES
# --------------------------
def base_feature_name(col: str) -> str:
    """
    Takes a one-hot encoded column name like 'Dst IP _172.21.0.120'
    and returns 'Dst IP'.
    """
    parts = col.split("_", 1)
    return parts[0].strip()

# Build mapping dict for replacement
feature_name_mapping = {f: base_feature_name(f) for f in feature_names}

# --------------------------
# FEATURE USAGE PER CLASS
# --------------------------
class_feature_usage = {}

for file_name in os.listdir(json_dir):
    if file_name.startswith("expl_") and file_name.endswith(".json"):
        json_path = os.path.join(json_dir, file_name)

        with open(json_path, 'r') as f:
            obj = json.load(f)

        usage = np.zeros(len(feature_names))

        for sample in obj:
            explanations = sample['explanations']
            for expl in explanations:
                for ind in expl:
                    usage[ind] += 1

        class_label = file_name[len("expl_"):-len(".json")].replace("_", " ")
        total_explanations = sum(len(sample['explanations']) for sample in obj)
        class_feature_usage[class_label] = usage / total_explanations

# --------------------------
# BUILD OUTPUT DF (CLEAN NAMES)
# --------------------------
feature_usage_df = pd.DataFrame(class_feature_usage, index=feature_names)

# Map to base names
feature_usage_df.index = feature_usage_df.index.map(lambda f: feature_name_mapping.get(f, f))

# If multiple one-hot columns collapse to the same base name, group & average
feature_usage_df = feature_usage_df.groupby(feature_usage_df.index).mean()

# --------------------------
# SAVE & PRINT
# --------------------------
top_features_per_class = {}
for class_label in feature_usage_df.columns:
    class_features = feature_usage_df[class_label].sort_values(ascending=False)
    top_features_per_class[class_label] = class_features

top_features_df = pd.DataFrame(top_features_per_class)
top_features_df.to_csv(output_file)

for class_label, features in top_features_per_class.items():
    print(f"\nTop 5 Features for {class_label}:")
    print(features.head(5))



"""
feature_names = df.drop(columns=['Binary_Label', 'Multiclass_Label']).columns.tolist()

class_feature_usage = {}

for file_name in os.listdir(json_dir):
    if file_name.startswith("expl_") and file_name.endswith(".json"):
        json_path = os.path.join(json_dir, file_name)
        
    
        with open(json_path, 'r') as f:
            obj = json.load(f)
        
        usage = np.zeros(len(feature_names))
        
        
        for sample in obj:  
            explanations = sample['explanations']
            for expl in explanations:
                for ind in expl:
                    usage[ind] += 1
        
       
        class_label = file_name[len("expl_"):-len(".json")].replace("_", " ")

        
        total_explanations = sum(len(sample['explanations']) for sample in obj)
        class_feature_usage[class_label] = usage / total_explanations

feature_usage_df = pd.DataFrame(class_feature_usage, index=feature_names)


top_features_per_class = {}
for class_label in feature_usage_df.columns:
    class_features = feature_usage_df[class_label].sort_values(ascending=False)
    top_features_per_class[class_label] = class_features

top_features_df = pd.DataFrame(top_features_per_class)
top_features_df.to_csv(output_file)

for class_label, features in top_features_per_class.items():
    print(f"\nTop 5 Features for {class_label}:")
    print(features)
"""