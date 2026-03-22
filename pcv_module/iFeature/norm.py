#!/usr/bin/env python
#_*_coding:utf-8_*_

import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalize_protein_features(input_dir, output_dir):

    # 初始化标准化器
    scaler = MinMaxScaler()

    # 遍历输入目录中的所有.npy文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".npy"):
            file_path = os.path.join(input_dir, filename)
            
            # 加载特征矩阵
            features = np.load(file_path, allow_pickle=True)
            features = features.reshape(-1, 1)
            # 进行Min-Max标准化
            normalized_features = scaler.fit_transform(features)
            
            # 保存标准化后的特征到输出目录
            output_path = os.path.join(output_dir, filename)
            np.save(output_path, normalized_features)
            print(f"proceeded: {filename}")

    print(f"{output_dir}")

# 示例使用
input_directory = "/home/inspur/marshall/iFu/iFeature/total/GDPC2/"# 替换为实际输入文件夹路径
output_directory ="/home/inspur/marshall/iFu/iFeature/total/GDPC_min2/"# 替换为实际输出文件夹路径

normalize_protein_features(input_directory, output_directory)