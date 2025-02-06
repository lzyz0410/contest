import os
import sys
third_packages = r"G:\\pyhton3119\\Lib\\site-packages"
sys.path.append(third_packages)

import ansa
from ansa import *
import numpy as np
import time
import pandas as pd
from utils_data import *
from utils_node import *
from utils_env import *
# 全局字典
node_id_to_transformed_coords = {}
from pathlib import Path
input_file = find_file_in_parents("output.k")
output_file = input_file.with_name(input_file.stem + "_transformed.k")
shell_property_file = find_file_in_parents("shell_property.csv")
file_lines = None



# 初始化函数：加载表面节点数据和目标坐标
def initialize_shell_data():
    global all_shell_points, node_id_to_transformed_coords, file_lines
    # 1. 获取表面节点和原始坐标
    nodes_method, nodes_param, range = "csv", shell_property_file, "A2:B35"
    all_shell_nodes = get_all_nodes(nodes_method, nodes_param, range)
    all_shell_points = np.array([[node._id] + list(node.position) for node in all_shell_nodes])

    # 2. 定义 shell_node_data 数组
    num_shell_nodes = len(all_shell_nodes)
    shell_node_data = np.empty((num_shell_nodes, 7), dtype=object)
    shell_node_data[:, 0] = [node._id for node in all_shell_nodes]
    shell_node_data[:, 1:4] = [node.position for node in all_shell_nodes]

    # 3. 加载目标坐标
    target_nodeids = set(int(point[0]) for point in all_shell_points)
    target_shell_points, file_lines = read_node_coordinates(input_file, target_nodeids)

    # 创建目标节点字典
    target_node_dict = {int(node_data_point[0]): node_data_point[1:] for node_data_point in target_shell_points}

    # 4. 更新 shell_node_data
    for idx, node_id in enumerate(shell_node_data[:, 0]):
        if node_id in target_node_dict:
            shell_node_data[idx, 4:7] = target_node_dict[node_id]

    # 构建 node_id_to_transformed_coords 字典
    node_id_to_transformed_coords = {int(row[0]): row[4:7] for row in shell_node_data}

    # 返回所有初始化数据
    return all_shell_points, node_id_to_transformed_coords, file_lines

def project_and_transform_to_surface(all_shell_points, node_direction_map):
    """
    根据投影方向计算每个内部节点的投影外部点和变换后的内部点，并将每个方向的结果合并返回
    将原本分散的函数逻辑合并为一个函数
    """
    inner_locating_points = []  # 存储原始内部定位点
    outer_locating_points = []  # 存储原始外部定位点
    transformed_inner_locating_points = []  # 存储变换后的内部定位点
    transformed_outer_locating_points = []  # 存储变换后的外部定位点
    
    # 预先过滤并缓存外部点
    shell_points_by_direction = {
        'Xnegative': all_shell_points[all_shell_points[:, 1] <= 0],
        'Xpositive': all_shell_points[all_shell_points[:, 1] >= 0],
        'Ynegative': all_shell_points[all_shell_points[:, 2] <= 0],
        'Ypositive': all_shell_points[all_shell_points[:, 2] >= 0],
        'Znegative': all_shell_points[all_shell_points[:, 3] <= 0],
        'Zpositive': all_shell_points[all_shell_points[:, 3] >= 0]
    }
    # 处理不同方向的投影
    for direction, nodes in node_direction_map.items():
        node_ids = [node[0] for node in nodes]
        inner_locating_nodes = get_all_nodes("node", node_ids)
        inner_points = np.array([[node._id] + list(node.position) for node in inner_locating_nodes])

        outer_locating_points_direction = []
        transformed_outer_points_direction = []
        transformed_inner_points_direction = []

        # 获取候选外部点
        candidates = shell_points_by_direction.get(direction, np.array([]))

        if len(candidates) == 0:
            print(f"警告：未找到 {direction} 方向的候选外部点！")
            continue

        for loc_point in inner_points:
            loc_node_id, loc_x, loc_y, loc_z = loc_point

            if direction == "Xnegative":
                mask = all_shell_points[:, 1] < loc_x
                dist_axis = (2, 3)  # 选择 YZ 方向最近的点
            elif direction == "Xpositive":
                mask = all_shell_points[:, 1] > loc_x
                dist_axis = (2, 3)
            elif direction == "Ynegative":
                mask = all_shell_points[:, 2] < loc_y
                dist_axis = (1, 3)  # 选择 XZ 方向最近的点
            elif direction == "Ypositive":
                mask = all_shell_points[:, 2] > loc_y
                dist_axis = (1, 3)
            elif direction == "Znegative":
                mask = all_shell_points[:, 3] < loc_z
                dist_axis = (1, 2)  # 选择 XY 方向最近的点
            elif direction == "Zpositive":
                mask = all_shell_points[:, 3] > loc_z
                dist_axis = (1, 2)
            else:
                raise ValueError(f"未知方向: {direction}")

            candidates = all_shell_points[mask]

            if len(candidates) == 0:
                # 若没有符合方向的点，选择最近点作为 fallback
                distances = np.linalg.norm(all_shell_points[:, 1:] - [loc_x, loc_y, loc_z], axis=1)
                nearest = all_shell_points[np.argmin(distances)]
            else:
                # 计算在选定的轴方向（YZ / XZ / XY）上的最小距离
                distances = np.linalg.norm(candidates[:, dist_axis] - loc_point[list(dist_axis)], axis=1)
                nearest = candidates[np.argmin(distances)]

            outer_locating_points_direction.append([nearest[0], nearest[1], nearest[2], nearest[3]])

        # 获取变换后的外部点
        transformed_outer_points_direction = get_target_coordinates(np.array(outer_locating_points_direction))

        # 计算变换后的内部点
        for inner, outer ,transformed_outer in zip(inner_points, outer_locating_points_direction, transformed_outer_points_direction):
            loc_node_id, loc_x, loc_y, loc_z = inner
            outer_node_id, outer_x, outer_y, outer_z = outer
            transformed_outer_node_id, transformed_outer_x, transformed_outer_y, transformed_outer_z = transformed_outer
            if direction in ["Xnegative", "Xpositive"]:
                proportion_x = (transformed_outer_x - outer_x) / (loc_x - outer_x)  
                transformed_x = loc_x + proportion_x * (loc_x - outer_x)
                transformed_inner_points_direction.append([loc_node_id, transformed_x, transformed_outer_y, transformed_outer_z])
            elif direction in ["Ynegative", "Ypositive"]:
                proportion_y = (transformed_outer_y - outer_y) / (loc_y - outer_y) 
                transformed_y = loc_y + proportion_y * (loc_y - outer_y)
                transformed_inner_points_direction.append([loc_node_id, transformed_outer_x, transformed_y, transformed_outer_z])
            elif direction in ["Znegative", "Zpositive"]:
                proportion_z = (transformed_outer_z - outer_z) / (loc_z - outer_z) 
                transformed_z = loc_z + proportion_z * (loc_z - outer_z)
                transformed_inner_points_direction.append([loc_node_id, transformed_outer_x, transformed_outer_y, transformed_z])

        transformed_outer_locating_points.extend(transformed_outer_points_direction)
        transformed_inner_locating_points.extend(transformed_inner_points_direction)
        inner_locating_points.extend(inner_points.tolist())
        outer_locating_points.extend(outer_locating_points_direction)

    return np.array(inner_locating_points), np.array(outer_locating_points), np.array(transformed_outer_locating_points), np.array(transformed_inner_locating_points)


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

def modify_transformed_inner_points(transformed_inner_locating_points, node_shifts):
    """
    只对特定的 node_id 进行不同方向的坐标平移，并返回修改后的 transformed_inner_locating_points。

    :param transformed_inner_locating_points: (N, 4) 数组，每个点 [node_id, x, y, z]
    :param node_shifts: 列表，包含多个 (node_id, shift_direction, shift_amount)
    :return: 修改后的 transformed_inner_locating_points
    """
    modified_points = np.copy(transformed_inner_locating_points)  # 复制数据，避免修改原始数据

    # 转换成字典，便于快速查找 (node_id: (shift_direction, shift_amount))
    node_shift_dict = {node_id: (shift_direction, shift_amount) for node_id, shift_direction, shift_amount in node_shifts}

    for i, point in enumerate(modified_points):
        node_id, x, y, z = point

        if int(node_id) in node_shift_dict:  # 仅修改指定的 node_id
            shift_direction, shift_amount = node_shift_dict[int(node_id)]
            
            if shift_direction == "X":
                modified_points[i][1] += shift_amount  # X 方向平移
            elif shift_direction == "Y":
                modified_points[i][2] += shift_amount  # Y 方向平移
            elif shift_direction == "Z":
                modified_points[i][3] += shift_amount  # Z 方向平移

    return modified_points
