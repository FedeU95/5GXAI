import subprocess
import os

# Paths
vote_explain_bin = "/home/federica/testbed_proto/xAI_container/vote-xai/src/vote_explain"             # vote-xai path
model_path = "PFCP.json"             # Path to the VoTE model
#model_path = "MSA.json" 
input_dir = "TP_PFCP" 
#output_dir = "./json/MSA_json"                   # Where to save .json files
output_dir = "./json/PFCP"  
os.makedirs(output_dir, exist_ok=True)


# Attack CSVs to process
attack_files = [
    f for f in os.listdir(input_dir)
    if f.endswith('.csv')
]

# Loop through each file and run vote_explain
for csv_file in attack_files:
    csv_path = os.path.join(input_dir, csv_file)
    output_name = os.path.splitext(csv_file)[0] + ".json"
    output_path = os.path.join(output_dir, output_name)

    cmd = [
        vote_explain_bin,
        "-m", model_path,
        csv_path,
        "--all",
        #"-T", "3600",  
        "-o", output_path
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Output or handle result
    if result.returncode == 0:
        print(f" Successfully generated: {output_path}")
    else:
        print(f" Failed on: {csv_file}")
        print(result.stderr)
