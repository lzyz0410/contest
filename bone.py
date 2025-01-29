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
from utils_rbf_transform import *

# 全局字典
node_id_to_transformed_coords = {}

def project_to_surface_along_x(inner_locating_points, all_shell_points):
    """
    将 inner_locating_points 沿着 X 负方向投影到 all_shell_points 所代表的表面。
    """
    outer_locating_points = []
    all_shell_points_x = all_shell_points[:, 1:]

    for loc_point in inner_locating_points:
        loc_x, loc_y, loc_z = loc_point[1], loc_point[2], loc_point[3]

        # 筛选沿 X 负方向的候选点（X <= loc_x）
        candidates = all_shell_points[all_shell_points[:, 1] <= loc_x]

        if len(candidates) == 0:
            nearest = all_shell_points[np.argmin(np.linalg.norm(all_shell_points_x - [loc_x, loc_y, loc_z], axis=1))]
        else:
            nearest = candidates[np.argmin(np.linalg.norm(candidates[:, 2:] - [loc_y, loc_z], axis=1))]

        outer_locating_points.append([nearest[0], nearest[1], loc_y, loc_z])

    return np.array(outer_locating_points)

def transform_inner_using_proportions(inner_locating_points, source_control_points, target_control_points):
    """
    使用内外定位点的比例关系来推算变换后的内定位点坐标。
    """
    # 打印 all_points 所有点的 ID 和坐标
    print("所有 all_points 原始内部点 (ID + X, Y, Z):")
    for points in inner_locating_points:
        print(f"ID: {int(points[0])}, X: {points[1]:.4f}, Y: {points[2]:.4f}, Z: {points[3]:.4f}")
    
    # 打印 source_control_points 所有点的 ID 和坐标
    print("\n所有 source_control_points 原始外部点(ID + X, Y, Z):")
    for points in source_control_points:
        print(f"ID: {int(points[0])}, X: {points[1]:.4f}, Y: {points[2]:.4f}, Z: {points[3]:.4f}")
    
    # 打印 target_control_points 所有点的 ID 和坐标
    print("\n所有 target_control_points 目标外部点(ID + X, Y, Z):")
    for points in target_control_points:
        print(f"ID: {int(points[0])}, X: {points[1]:.4f}, Y: {points[2]:.4f}, Z: {points[3]:.4f}")

    proportions = (target_control_points[:, 1] - source_control_points[:, 1]) / (inner_locating_points[:, 1] - source_control_points[:, 1])
    proportions[proportions == np.inf] = 0  # 处理除零情况

    transformed_inner_points = np.array([
        [inner[0], inner[1] + proportions[i] * (inner[1] - source_control_points[i][1]), target_control_points[i][2], target_control_points[i][3]]
        for i, inner in enumerate(inner_locating_points)
    ])

    return transformed_inner_points

def get_target_coordinates(points):
    """
    根据给定的 points 数组，从 shell_node_data 中查找每个节点的变换后坐标，并构建新的数组。

    :param points: 包含节点 ID 和坐标的 numpy 数组，形状为 (n, 4)，每行格式为 [id, x, y, z]
    :return: 包含节点 ID 和变换后坐标的 numpy 数组，形状为 (n, 4)
    """
    target_points = []

    for point in points:
        node_id = int(point[0])  # 获取输入点的 ID
        if node_id in node_id_to_transformed_coords:
            # 获取变换后的坐标
            transformed_coords = node_id_to_transformed_coords[node_id]
            target_points.append([node_id] + list(transformed_coords))
        else:
            print(f"警告：未找到节点 ID {node_id} 的变换坐标。")

    return np.array(target_points, dtype=float)

def transform_target_points(target_select_points, transformed_inner_locating_points):
    """
    将 transformed_inner_locating_points 添加到 target_select_points 中
    假设 target_select_points 和 transformed_inner_locating_points 的维度可以匹配，
    但 target_select_points 长度更长。
    """
    if not isinstance(target_select_points, np.ndarray):
        raise TypeError("target_select_points 必须是 numpy 数组")
    if not isinstance(transformed_inner_locating_points, np.ndarray):
        raise TypeError("transformed_inner_locating_points 必须是 numpy 数组")

    target_control_points = []

    # 使用 numpy 的 arange 函数替代 range
    for i in np.arange(target_select_points.shape[0]):
        target_select_point = target_select_points[i]
        transformed_inner_point = transformed_inner_locating_points[i % transformed_inner_locating_points.shape[0]]

        if target_select_point.shape[0] != transformed_inner_point.shape[0]:
            raise ValueError("目标点和变换后的内部定位点的长度不匹配")

        new_point = target_select_point + transformed_inner_point
        target_control_points.append(new_point)

    return np.array(target_control_points)

start_time = time.time()

# 获取表面节点和原始坐标
nodes_method, nodes_param, range = "csv", "E:\\LZYZ\\Scoliosis\\RBF\\Contest\\final\\shell_property.csv", "A2:B35"
all_shell_nodes = get_all_nodes(nodes_method, nodes_param, range)
all_shell_points = np.array([[node._id] + list(node.position) for node in all_shell_nodes])

# 定义 shell_node_data 数组
num_shell_nodes = len(all_shell_nodes)
shell_node_data = np.empty((num_shell_nodes, 7), dtype=object)
shell_node_data[:, 0] = [node._id for node in all_shell_nodes]
shell_node_data[:, 1:4] = [node.position for node in all_shell_nodes]
# 直接在主函数中将 shell_node_data 转换为字典，字典的键为节点 ID，值为变换后的坐标 (X', Y', Z')
node_id_to_transformed_coords = {int(row[0]): row[4:7] for row in shell_node_data}

# 提取 all_shell_points 中的 node IDs，并转换为集合，表面目标坐标
input_file = r"E:\\LZYZ\\Scoliosis\\RBF\\Contest\\final\\output.k"
target_nodeids = set(int(point[0]) for point in all_shell_points)
target_shell_points, file_lines = read_node_coordinates(input_file, target_nodeids)

# 创建目标节点字典
target_node_dict = {int(node_data_point[0]): node_data_point[1:] for node_data_point in target_shell_points}
# 通过 node_id 将目标坐标批量填充到 shell_node_data 中
for idx, node_id in enumerate(shell_node_data[:, 0]):
    if node_id in target_node_dict:
        shell_node_data[idx, 4:7] = target_node_dict[node_id]

# 内部定位节点映射
node_name_map = {
    89063944: 'T12',  
    89059500: 'T1',
    89004125: 'Manubrium',
    89003192: 'xiphisternum',
    89049118: 'T2',
    89059890: 'T3',
    89060409: 'T4',
    89060803: 'T5',
    89061235: 'T6',
    89062096: 'T7',
    89062148: 'T8',
    89062189: 'T9',
    89063196: 'T10',
    89051426: 'T11',
}
# 提取 node_name_map 中的节点 ID
node_ids = list(node_name_map.keys())
inner_locating_nodes = get_all_nodes("node", node_ids)
inner_locating_points = np.array([[node._id] + list(node.position) for node in inner_locating_nodes])  # 将 nodeid 转换为整数
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

transformed_inner_locating_points = transform_inner_using_proportions(
    inner_locating_points,  # 需要变换的内部定位点
    outer_locating_points,  # 原始外部定位点
    get_target_coordinates(outer_locating_points),  # 变形后的外部定位点
)

# 输出推算后的内部点坐标
print("\n推算后的内部点坐标 (ID + X, Y, Z):")
for point in transformed_inner_locating_points:
    print(f"ID: {int(point[0])}, X: {point[1]:.4f}, Y: {point[2]:.4f}, Z: {point[3]:.4f}")

transform_nodes = get_all_nodes("set", ["101"])
transform_points = np.array([[node._id] + list(node.position) for node in transform_nodes])  # 将 nodeid 转换为整数
source_shell_nodes = get_all_nodes("pid", ["89200801","89700801"])
source_select_nodes = select_uniform_nodes(source_shell_nodes, 100)
source_select_points = np.array([[node._id] + list(node.position) for node in source_select_nodes])  # 将 nodeid 转换为整数
target_select_points = get_target_coordinates(source_select_points)

# 在调用 transform_target_points 之前打印类型，确认输入是否正确
source_control_points = np.vstack([source_select_points, inner_locating_points])
target_control_points = np.vstack([target_select_points, transformed_inner_locating_points])
print("source_control_points 的数量:", source_control_points.shape[0])
print("target_control_points 的数量:", target_control_points.shape[0])

# print([int(point[0]) for point in source_control_points]) # 提取 ID 并转换为整数
# print([int(point[0]) for point in target_control_points]) # 提取 ID 并转换为整数


transformed_points = rbf_transform_3d_chunked(
    transform_points,
    source_control_points,                
    target_control_points,
    0)

write_modified_coordinates(
    output_file=input_file.replace(".k", "_modi.k"),
    file_lines=file_lines,
    updated_node_data=transformed_points
)
print("修改后的坐标已写入文件:", input_file.replace(".k", "_modi.k"))

end_time = time.time()
print(f"文件名: {os.path.basename(__file__)}, 总运行时间: {end_time - start_time:.2f} s")