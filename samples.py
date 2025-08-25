import pandas as pd
import os

# Read data
df = pd.read_csv('true_positives_balanced_PFCP.csv')
#df = pd.read_csv('false_negatives_all_5GNIDD.csv')

label_col = df.columns[-1]

# Folder where you want to save
output_folder = "TP_PFCP"

# Create folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate over each unique class label
for label in df[label_col].unique():
    # Filter rows belonging to the current label
    class_df = df[df[label_col] == label]
    
    # Drop the last two columns
    class_df = class_df.iloc[:, :-2]
    
    # Sanitize label for filename (replace spaces with underscores)
    label_str = str(label).replace(' ', '_')
    
    # Save to CSV without header or index in the target folder
    output_path = os.path.join(output_folder, f'class_{label_str}.csv')
    class_df.to_csv(output_path, index=False, header=False)
