import os
import sys
third_packages = r"G:\\anaconda3\\envs\\ansa_meta_env\\Lib\\site-packages"
sys.path.append(third_packages)
import re
import ansa
from ansa import *
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import time

# 获取指定 Set ID 内的所有节点
def get_nodes_from_set(set_ids):
    all_nodes = set()  # 使用集合来存储节点，确保唯一性
    for set_id in set_ids:
        set_entity = base.GetEntity(constants.LSDYNA, 'SET', set_id)
        if set_entity:
            nodes = base.CollectEntities(constants.LSDYNA, set_entity, 'NODE', recursive=False)
            if nodes:
                all_nodes.update(nodes)
                print(f"Set ID {set_id} 中找到 {len(nodes)} 个节点。")
            else:
                print(f"Set ID {set_id} 中没有找到任何节点。")
        else:
            print(f"无法找到 Set ID: {set_id}")
    return list(all_nodes)

# 获取指定节点 ID 的坐标
def get_nodes_from_ids(node_ids):
    nodes = []
    coords = []
    for node_id in node_ids:
        node = base.GetEntity(constants.LSDYNA, 'NODE', node_id)
        if node:
            nodes.append(node)
            coords.append(node.position)
        else:
            print(f"无法找到节点 ID: {node_id}")
    return nodes, np.array(coords)

# 确保所有用于线性代数计算的数组是 numpy.float64 类型
def rbf_transform_3d_chunked(all_coords, source_control_coords, target_control_coords, alpha, chunk_size=20000):
    """
    使用 RBF 插值对目标点进行变换，支持分块计算。
    参数：
    - all_coords: 需要变换的目标点坐标 (N x 3)
    - source_control_coords: 源控制点坐标 (M x 3)
    - target_control_coords: 目标控制点坐标 (M x 3)
    - alpha: 正则化参数。
    - chunk_size: 每次处理的目标点数量。

    返回：
    - 变换后的目标点坐标 (N x 3)
    """
    def rbf_phi(r):
        return r**2 * np.log(r + 1e-8)  # 添加 1e-8 防止 log(0)

    # 1. 计算 RBF 权重和线性参数
    N = len(source_control_coords)
    dist_matrix = np.sqrt(((source_control_coords[:, None, :] - source_control_coords[None, :, :])**2).sum(axis=2))
    Phi = rbf_phi(dist_matrix) + alpha * np.identity(N)
    P = np.column_stack([np.ones(N), source_control_coords])
    M = np.block([[Phi, P], [P.T, np.zeros((P.shape[1], P.shape[1]))]])
    D = np.vstack([target_control_coords, np.zeros((P.shape[1], target_control_coords.shape[1]))])
    W = np.linalg.solve(M, D)
    weights, linear_params = W[:-P.shape[1]], W[-P.shape[1]:]

    # 2. 分块计算目标点
    transformed_coords = np.empty_like(all_coords)
    for start in range(0, len(all_coords), chunk_size):
        end = min(start + chunk_size, len(all_coords))
        chunk = all_coords[start:end]

        # 计算当前块的距离矩阵
        dist_matrix = np.linalg.norm(chunk[:, None, :] - source_control_coords[None, :, :], axis=2)
        Phi_chunk = rbf_phi(dist_matrix)
        P_chunk = np.column_stack([np.ones(len(chunk)), chunk])

        # 插值变换
        transformed_coords[start:end] = Phi_chunk @ weights + P_chunk @ linear_params

    return transformed_coords


# 从 K 文件中根据 Node ID 找变换后的坐标
def read_coordinates_from_k_file(input_file, target_node_ids):
    with open(input_file, 'r') as f:
        file_lines = f.readlines()

    target_node_ids = set(map(str, target_node_ids))  # 转为字符串集合，便于匹配
    node_data = []
    for line in file_lines:
        if not line.strip() or line.startswith('$'):
            continue
        match = re.match(r'\s*(\d+)\s+([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)', line)
        if match:
            node_id = match.group(1)
            if node_id in target_node_ids:
                coords = list(map(float, match.groups()[1:]))
                node_data.append([int(node_id)] + coords)
    return {int(data[0]): np.array(data[1:], dtype=np.float64) for data in node_data}

def batch_process_transform(input_file, transformations):
    for idx, transformation in enumerate(transformations):
        set_id_to_transform = transformation["set_id_to_transform"]
        set_id_control = transformation["set_id_control"]
        num_sample_points = transformation["num_sample_points"]
        alpha = transformation.get("alpha", 0.01)

        print(f"\n开始处理第 {idx + 1} 个变换：Set {set_id_to_transform} -> Set {set_id_control}，Alpha: {alpha}", flush=True)

        try:
            # 从 ANSA 读取 Set ID 的节点
            bony_nodes = get_nodes_from_set([set_id_to_transform])
            bony_node_ids = [node._id for node in bony_nodes]
            bony_node_coords = np.array([node.position for node in bony_nodes], dtype=np.float64)
            print(f"Set {set_id_to_transform} 中的节点数量: {len(bony_node_ids)}", flush=True)

            # 从 ANSA 读取控制点的节点
            if isinstance(set_id_control, list):
                control_nodes = get_nodes_from_set(set_id_control)
            else:
                control_nodes = get_nodes_from_set([set_id_control])

            control_node_ids = [node._id for node in control_nodes]
            source_control_coords = np.array([node.position for node in control_nodes], dtype=np.float64)
            print(f"Set {set_id_control} 中的控制点数量: {len(control_node_ids)}", flush=True)

            # 均匀选取控制点
            sampled_indices = np.linspace(0, len(control_node_ids) - 1, num_sample_points, dtype=int)
            sampled_control_ids = [control_node_ids[i] for i in sampled_indices]
            sampled_control_coords = source_control_coords[sampled_indices]

            print(f"均匀选取的 {num_sample_points} 个控制点 ID 和坐标：")
            for i, (node_id, coord) in enumerate(zip(sampled_control_ids, sampled_control_coords)):
                print(f"{i + 1}: Node ID: {node_id}, Coordinates: {coord.tolist()}", flush=True)

            # 从 K 文件中读取控制点的变换后坐标
            k_control_coords = read_coordinates_from_k_file(input_file, control_node_ids)
            target_control_coords = np.array([k_control_coords[node_id] for node_id in control_node_ids])
            print(f"控制点变换后坐标（前 5 个）：\n{target_control_coords[:5]}", flush=True)

            # 对 Set 的节点应用 RBF 变换
            print(f"开始对 Set {set_id_to_transform} 的节点应用 RBF 变换...", flush=True)
            transformed_bony_coords = rbf_transform_3d_chunked(
                bony_node_coords, source_control_coords, target_control_coords, alpha)

            # 更新 ANSA 中的 Set ID 的节点坐标
            print(f"更新 Set {set_id_to_transform} 中的节点坐标...", flush=True)
            for node, new_coords in zip(bony_nodes, transformed_bony_coords):
                node.position = new_coords

            print(f"Set {set_id_to_transform} 的节点坐标已更新。\n", flush=True)

        except Exception as e:
            print(f"处理 Set {set_id_to_transform} 时发生错误：{e}", flush=True)


# 主程序入口
if __name__ == "__main__":
    input_file = r"E:\LZYZ\Scoliosis\RBF\Contest\upload\after_test567_ansaSmoothANDreflect.k"

    # 定义批量运行的变换设置
transformations = [
    {"set_id_to_transform": 81, "set_id_control": 31, "num_sample_points": 50, "alpha": 0},  # 脚R
    {"set_id_to_transform": 82, "set_id_control": 32, "num_sample_points": 50, "alpha": 0},  # 小腿R
    {"set_id_to_transform": 83, "set_id_control": 33, "num_sample_points": 50, "alpha": 0},  # 膝盖R
    {"set_id_to_transform": 84, "set_id_control": 34, "num_sample_points": 50, "alpha": 0},  # 大腿R
    {"set_id_to_transform": 85, "set_id_control": 35, "num_sample_points": 50, "alpha": 0},  # 臀R
    {"set_id_to_transform": 87, "set_id_control": 86, "num_sample_points": 50, "alpha": 0},  # 脚L
    {"set_id_to_transform": 89, "set_id_control": 88, "num_sample_points": 50, "alpha": 0},   # 小腿L
    {"set_id_to_transform": 91, "set_id_control": 90, "num_sample_points": 50, "alpha": 0},   # kneeL
    {"set_id_to_transform": 93, "set_id_control": 92, "num_sample_points": 50, "alpha": 0},   # 大腿L
    {"set_id_to_transform": 94, "set_id_control": 42, "num_sample_points": 50, "alpha": 0},   # 臀L

    # {"set_id_to_transform": 96, "set_id_control": 97, "num_sample_points": 50, "alpha": 0},   # 胸Lx
    # {"set_id_to_transform": 98, "set_id_control": 99, "num_sample_points": 50, "alpha": 0},   # 胸Rx
    #{"set_id_to_transform": 95, "set_id_control": 19, "num_sample_points": 50, "alpha": 0},   # 胸all用这个x
    #{"set_id_to_transform": 100, "set_id_control": [18,17], "num_sample_points": 50, "alpha": 0},   # t头颈all用这个x
    {"set_id_to_transform": 101, "set_id_control": [19,18,17], "num_sample_points": 150, "alpha": 0},   # toujingxiong
    #{"set_id_to_transform": 102, "set_id_control": 16, "num_sample_points": 50, "alpha": 0},   # 左手x
    #{"set_id_to_transform": 103, "set_id_control": 15, "num_sample_points": 50, "alpha": 0},   # 右手x
    {"set_id_to_transform": 104, "set_id_control": 105, "num_sample_points": 50, "alpha": 0},   # 左肩
    {"set_id_to_transform": 106, "set_id_control": 107, "num_sample_points": 50, "alpha": 0},   # 右肩
    {"set_id_to_transform": 108, "set_id_control": 110, "num_sample_points": 50, "alpha": 0},   # 左臂
    {"set_id_to_transform": 109, "set_id_control": 111, "num_sample_points": 50, "alpha": 0},   # 右臂
]
# 记录程序的起始时间
total_start_time = time.time()
print("开始批量运行变换...")

# 批量运行
batch_process_transform(input_file, transformations)

# 记录程序结束时间
total_end_time = time.time()
total_duration = total_end_time - total_start_time
print(f"批量运行完成，总耗时: {total_duration:.2f} 秒")
