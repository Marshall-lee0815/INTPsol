#!/usr/bin/env python
#_*_coding:utf-8_*_

import os
import numpy as np

def concatenate_features(directories, output_dir):

    # 检查输出目录是否存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有目录中的文件名列表（假设每个目录中的文件名相同）
    file_names = set(os.listdir(directories[0]))  # 假设第一个目录包含所有文件
    for dir_path in directories[1:]:
        file_names &= set(os.listdir(dir_path))  # 只保留各个目录中都有的文件名

    # 遍历所有匹配的文件名
    for file_name in file_names:
        # 用于存储当前文件的所有特征
        all_features = []

        # 遍历每个目录，将相同文件名的.npy文件加载并连接
        for dir_path in directories:
            file_path = os.path.join(dir_path, file_name)
            
            # 加载文件并确保是一个numpy数组
            features = np.load(file_path)
            all_features.append(features)

        # 将所有特征连接起来
        concatenated_features = np.concatenate(all_features, axis=0)

        # 保存连接后的特征到输出目录
        output_file_path = os.path.join(output_dir, file_name)
        np.save(output_file_path, concatenated_features)
        print(f"chuli: {file_name}")

    print(f"chuliwancheng: {output_dir}")

# 示例使用
directories = [
    "/home/inspur/marshall/iFu/iFeature/total/APAAC_min2/", 
    "/home/inspur/marshall/iFu/iFeature/total/CTDC_min2/",
    "/home/inspur/marshall/iFu/iFeature/total/CTDT_min2/",
    "/home/inspur/marshall/iFu/iFeature/total/GAAC_min2/",
    "/home/inspur/marshall/iFu/iFeature/total/GDPC_min2/",
]

output_directory = "/home/inspur/marshall/iFu/iFeature/total/total3/" # 输出目录路径

concatenate_features(directories, output_directory)