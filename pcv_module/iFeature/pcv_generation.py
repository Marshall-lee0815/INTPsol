import os
import numpy as np
import pandas as pd

fasta_folder = "/home/inspur/marshall/dataset/resultsp/augfasta/"
save_base_folder = "/home/inspur/marshall/iFu/iFeature/total"
ifeature_path = "/home/inspur/marshall/iFu/iFeature/iFeature.py"
feature_types = ["APAAC", "CTDT", "CTDC", "GDPC", "GAAC"]

for feature_type in feature_types:
    feature_folder = os.path.join(save_base_folder, feature_type)
    os.makedirs(feature_folder, exist_ok=True)

    for fasta_file in os.listdir(fasta_folder):
        if fasta_file.endswith(".fasta"):
            fasta_path = os.path.join(fasta_folder, fasta_file)
            protein_id = os.path.splitext(fasta_file)[0]
            output_path = os.path.join(feature_folder, f"{protein_id}.txt")

            cmd = f"python {ifeature_path} --file {fasta_path} --type {feature_type} --out {output_path}"
            os.system(cmd)
            print(f"Saved {feature_type} for {protein_id} to {output_path}")