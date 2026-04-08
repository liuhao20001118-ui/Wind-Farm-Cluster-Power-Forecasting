'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2026-04-07 12:05:18
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2026-04-07 12:26:34
FilePath: /liuhao/My物理机理更合理_copy_2_物理约束/utils/noise_utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import numpy as np
import torch

def add_white_noise(data, noise_level):

    if noise_level == 0:
        return data
    std = np.std(data)
    noise = np.random.normal(0, noise_level * std, data.shape)
    return data + noise

def add_random_drop(data, noise_level):

    if noise_level == 0:
        return data
    data_flat = data.flatten()
    num_to_drop = int(len(data_flat) * noise_level)
    indices_to_drop = np.random.choice(len(data_flat), num_to_drop, replace=False)
    data_flat[indices_to_drop] = 0
    return data_flat.reshape(data.shape)

def add_structured_drop(data, noise_level, avg_chunk_len=10):

    if noise_level == 0:
        return data
    
    # data shape is (timesteps, num_nodes, num_features)
    timesteps, num_nodes, num_features = data.shape
    total_elements = timesteps * num_nodes * num_features
    num_to_drop = int(total_elements * noise_level)
    
    # 在时间维度上进行丢弃
    noisy_data = data.copy()
    num_timesteps_to_drop = int(timesteps * noise_level)
    
    if num_timesteps_to_drop == 0:
        return data

    # 随机选择起始点
    start_indices = np.random.choice(timesteps - 1, size=num_timesteps_to_drop, replace=True)
    for start in start_indices:
         # 随机选择一个受影响的节点和特征进行丢弃，增加真实性
        node_to_drop = np.random.randint(0, num_nodes)
        feature_to_drop = np.random.randint(0, num_features)
        noisy_data[start, node_to_drop, feature_to_drop] = 0
        
    return noisy_data


def add_composite_noise(data, noise_level):

    if noise_level == 0:
        return data
    
    # 1. 添加白噪声
    noisy_data = add_white_noise(data, noise_level / 3)
    # 2. 添加随机丢弃
    noisy_data = add_random_drop(noisy_data, noise_level / 3)
    # 3. 添加结构化丢弃
    noisy_data = add_structured_drop(noisy_data, noise_level / 3)
    
    return noisy_data