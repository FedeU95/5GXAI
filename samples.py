import pandas as pd

df = pd.read_csv('true_positives_balanced_MSA.csv')

label_col = df.columns[-1]

# Iterate over each unique class label
for label in df[label_col].unique():
    # Filter rows belonging to the current label
    class_df = df[df[label_col] == label]
    
    # Drop the last two columns
    class_df = class_df.iloc[:, :-2]
    
    # Sanitize label for filename (replace spaces with underscores)
    label_str = str(label).replace(' ', '_')
    
    # Save to CSV without header or index
    class_df.to_csv(f'class_{label_str}.csv', index=False, header=False)