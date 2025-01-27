import os
import sys
third_packages = r"G:\\anaconda3\\envs\\ansa_meta_env\\Lib\\site-packages"
sys.path.append(third_packages)

import ansa
from ansa import *
import numpy as np
import time
import pandas as pd
from utils_data import *
from utils_node import *

def project_to_surface_along_x(inner_locating_points, all_shell_points):
    """
    将 inner_locating_points 沿着 X 负方向投影到 all_shell_points 所代表的表面。

    inner_locating_points: 需要投影的点集，形如 (N, 4) 的数组（ID + X, Y, Z）
    all_shell_points: 表面上的点集，形如 (M, 4) 的数组（ID + X, Y, Z）

    返回: 投影后的点集，形如 (N, 4) 的数组（投影点ID + X_proj, Y_proj, Z_proj）
    """
    outer_locating_points = []

    for loc_point in inner_locating_points:
        loc_node_id, loc_x, loc_y, loc_z = loc_point

        # 筛选沿 X 负方向的候选点（X <= loc_x）
        mask = all_shell_points[:, 1] <= loc_x
        candidates = all_shell_points[mask]

        if len(candidates) == 0:
            # 如果没有候选点，直接使用最近点（不考虑方向）
            distances = np.linalg.norm(all_shell_points[:, 1:] - [loc_x, loc_y, loc_z], axis=1)
            min_idx = np.argmin(distances)
            nearest = all_shell_points[min_idx]
        else:
            # 在候选点中找到 Y/Z 最近的
            distances = np.linalg.norm(candidates[:, 2:] - [loc_y, loc_z], axis=1)
            min_idx = np.argmin(distances)
            nearest = candidates[min_idx]

        # 投影点的 ID 为皮肤表面点的 ID（转换为整数），X/Y/Z 坐标为投影点的坐标
        projected_point = [int(nearest[0]), nearest[1], loc_y, loc_z]
        outer_locating_points.append(projected_point)

    return np.array(outer_locating_points)

def transform_inner_using_proportions(inner_locating_points, source_control_points, target_control_points):
    """
    使用内外定位点的比例关系来推算变换后的内定位点坐标。
    只对 X 轴进行比例变化，Y 和 Z 坐标保持与变换后的外部定位点一致。

    inner_locating_points: 内部定位点，形如 (N, 4) 数组（ID, X, Y, Z）
    source_control_points: 外部定位点，变换前，形如 (N, 4) 数组（ID, X, Y, Z）
    target_control_points: 外部定位点，变换后，形如 (N, 4) 数组（ID, X_transformed, Y_transformed, Z_transformed）

    返回: 变换后的内定位点，形如 (N, 4) 数组（ID, X_transformed, Y_transformed, Z_transformed）
    """

    def calculate_x_proportions(inner_points, source_points, target_points):
        proportions = []
        for inner, source, target in zip(inner_points, source_points, target_points):
            # 计算比例：变换后的外部点X - 原始外部点X / 原始内部点X - 原始外部点X
            if (source[1] - inner[1]) == 0:  # 遇到源坐标的X和内部点X相同的情况，跳过
                print(f"Warning: Source X ({source[1]}) and inner X ({inner[1]}) are the same for ID {source[0]} - Skipping.")
                proportions.append(0)  # 遇到特殊情况，比例为0，避免除零
            else:
                x_ratio = (target[1] - source[1]) / (inner[1] - source[1])
                proportions.append(x_ratio)
        return np.array(proportions)

    def apply_proportions(inner_locating_points, proportions, target_control_points):
        """
        根据比例推算变换后的内定位点坐标。
        只变换 X 坐标，Y 和 Z 坐标直接使用目标外部定位点的 Y 和 Z 坐标。
        """
        transformed_inner_points = []
        for i, inner in enumerate(inner_locating_points):
            proportions_i = proportions[i]
            
            # 计算变换后的 X 坐标（只基于比例）
            transformed_x = inner[1] + proportions_i * (inner[1] - source_control_points[i][1])
            
            # 获取变换后的 Y 和 Z 坐标（直接使用目标外部定位点的 Y 和 Z 坐标）
            transformed_y = target_control_points[i][2]
            transformed_z = target_control_points[i][3]
            
            transformed_inner_points.append([inner[0], transformed_x, transformed_y, transformed_z])
        
        return np.array(transformed_inner_points)

    # 计算 X 轴的比例
    proportions = calculate_x_proportions(inner_locating_points, source_control_points, target_control_points)

    # 根据比例推算变换后的内定位点
    transformed_inner_points = apply_proportions(inner_locating_points, proportions, target_control_points)

    return transformed_inner_points


start_time = time.time()

# 根据配置获取目标节点
nodes_method = "csv"  # 获取节点的方式：从 CSV 获取 PID 列表
nodes_param = "E:\\LZYZ\\Scoliosis\\RBF\\Contest\\final\\shell_property.csv"
range = "A2:B35"  # 读取 PID 的范围
all_shell_nodes = get_all_nodes(nodes_method, nodes_param, range)
all_shell_points = np.array([[int(node._id)] + list(node.position) for node in all_shell_nodes])  # 将 nodeid 转换为整数

# 定义 NumPy 数组的结构，初始化为空数组
num_shell_nodes = len(all_shell_nodes)
# 使用 np.empty 创建一个空的 NumPy 数组，dtype=object 表示每个元素可以是任意类型
shell_node_data = np.empty((num_shell_nodes, 7), dtype=object)

# 填充原始坐标和目标坐标（目标坐标初始为 None）
shell_node_data[:, 0] = [int(node._id) for node in all_shell_nodes]  # 节点 ID（转换为整数）
shell_node_data[:, 1:4] = [node.position for node in all_shell_nodes]  # 原始坐标（X, Y, Z）
shell_node_data[:, 4:7] = None  # 目标坐标初始为 None (X, Y, Z)

# 提取 all_shell_points 中的 node IDs，并转换为集合
target_nodeids = set(int(point[0]) for point in all_shell_points)
input_file = r"E:\\LZYZ\\Scoliosis\\RBF\\Contest\\final\\output.k"
target_shell_points, file_lines = read_node_coordinates(input_file,target_nodeids)

# 创建字典映射：节点 ID -> 行索引（避免多次查找）
node_id_to_index = {int(node_id): idx for idx, node_id in enumerate(shell_node_data[:, 0])}  # 将 nodeid 转换为整数

# 将目标坐标批量填充到 shell_node_data 中
# 提取目标坐标点的 node_id 和坐标
target_node_dict = {int(node_data_point[0]): node_data_point[1:] for node_data_point in target_shell_points}  # 将 nodeid 转换为整数

# 通过 node_id 将目标坐标批量填充到 shell_node_data 中
for idx, node_id in enumerate(shell_node_data[:, 0]):
    if node_id in target_node_dict:
        shell_node_data[idx, 4:7] = target_node_dict[node_id]

# 内部定位节点映射
node_name_map = {
    89063944: 'T12',  # 将 nodeid 转换为整数
    89059500: 'T1',
    89004125: 'Manubrium',
    89003192: 'xiphisternum'
}
# 提取 node_name_map 中的节点 ID
node_ids = list(node_name_map.keys())
inner_locating_nodes = get_all_nodes("node", node_ids)
inner_locating_points = np.array([[int(node._id)] + list(node.position) for node in inner_locating_nodes])  # 将 nodeid 转换为整数
outer_locating_points = project_to_surface_along_x(inner_locating_points, all_shell_points)

# 提取 inner_locating_points 和 outer_locating_nodes 中的 ID
inner_ids = [int(point[0]) for point in inner_locating_points]  # 提取 ID 并转换为整数
outer_ids = [int(point[0]) for point in outer_locating_points]   # 提取 ID 并转换为整数

# 输出内部定位点的 ID 列表
print("内部定位点 ID 列表 (inner_locating_points):")
print(inner_ids)

# 输出投影后的定位点 ID 列表
print("\n投影后的定位点 ID 列表 (outer_locating_points):")
print(outer_ids)

# 从 shell_node_data 中提取 source_control_points 和 target_control_points
source_control_points = []  # 原始外部定位点
target_control_points = []  # 变形后的外部定位点

for outer_id in outer_ids:
    # 找到 outer_id 在 shell_node_data 中的索引
    idx = node_id_to_index[outer_id]
    
    # 提取原始坐标 (source_control_points)
    original_coords = shell_node_data[idx, 1:4]  # 原始坐标 (X, Y, Z)
    source_control_points.append([outer_id] + original_coords.tolist())
    
    # 提取目标坐标 (target_control_points)
    target_coords = shell_node_data[idx, 4:7]  # 目标坐标 (X', Y', Z')
    target_control_points.append([outer_id] + target_coords.tolist())

# 转换为 NumPy 数组
source_control_points = np.array(source_control_points, dtype=float)
target_control_points = np.array(target_control_points, dtype=float)

# 输出原始内部定位点坐标
print("\n原始内部定位点坐标 (ID + X, Y, Z):")
for point in inner_locating_points:
    print(f"ID: {int(point[0])}, X: {point[1]:.4f}, Y: {point[2]:.4f}, Z: {point[3]:.4f}")

# 输出原始外部定位点坐标
print("\n原始外部定位点坐标 (ID + X, Y, Z):")
for point in source_control_points:
    print(f"ID: {int(point[0])}, X: {point[1]:.4f}, Y: {point[2]:.4f}, Z: {point[3]:.4f}")

# 输出变形后的外部定位点坐标
print("\n变形后的外部定位点坐标 (ID + X', Y', Z'):")
for point in target_control_points:
    print(f"ID: {int(point[0])}, X': {point[1]:.4f}, Y': {point[2]:.4f}, Z': {point[3]:.4f}")    

transformed_inner_locating_points = transform_inner_using_proportions(
    inner_locating_points,  # 需要变换的内部定位点
    source_control_points,  # 原始外部定位点
    target_control_points,  # 变形后的外部定位点
)

# 输出推算后的内部点坐标
print("\n推算后的内部点坐标 (ID + X, Y, Z):")
for point in transformed_inner_locating_points:
    print(f"ID: {int(point[0])}, X: {point[1]:.4f}, Y: {point[2]:.4f}, Z: {point[3]:.4f}")


write_modified_coordinates(
    output_file=input_file.replace(".k", "_modi.k"),
    file_lines=file_lines,
    updated_node_data=transformed_inner_locating_points
)
print("修改后的坐标已写入文件:", input_file.replace(".k", "_modi.k"))

end_time = time.time()
print(f"文件名: {os.path.basename(__file__)}, 总运行时间: {end_time - start_time:.2f} s")