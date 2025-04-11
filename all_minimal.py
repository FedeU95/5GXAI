import os
import json
import numpy as np
import pandas as pd

# The function to compute all minimal with vote-xai did not have a python binding at the time of the writing
# The json files contain all minimal explanations per class computed using the vote-xai binary
# To compute all minimal: 
# in /src: ./vote_explain -m test-model.json test-samples.csv --all -o explanations.json

#json_dir = "./json"
json_dir = "./json/MSA_json"
output_file = "top_features_per_class_MSA_new.csv"

csv_path = 'true_positives_balanced_MSA.csv'
df = pd.read_csv(csv_path)
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
