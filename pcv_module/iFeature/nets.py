#!/usr/bin/env python
#_*_coding:utf-8_*_

import os
import numpy as np

def convert_txt_to_npy(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            try:
                data = []
                with open(input_path, 'r') as f:
                    for line in f:
                        if line.startswith("#"):
                            continue  
                        parts = line.strip().split()
                        if len(parts) <= 1:
                            continue  
                        values = list(map(float, parts[1:]))
                        data.append(values)

                array = np.array(data)
                protein_id = os.path.splitext(filename)[0]
                output_path = os.path.join(output_dir, f"{protein_id}.npy")
                np.save(output_path, array)
                print(f" {filename} 2 {protein_id}.npy")
            except Exception as e:
                print(f"{filename}, {e}")

    print(f"1{output_dir}")
    
convert_txt_to_npy(
    input_dir="/home/inspur/marshall/iFu/iFeature/total/GDPC/",
    output_dir="/home/inspur/marshall/iFu/iFeature/total/GDPC2/"
)