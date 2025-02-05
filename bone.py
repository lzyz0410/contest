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
# 全局变量，用于存储所有的变换后的节点
all_transformed_points = []
input_file = r"E:\\LZYZ\\Scoliosis\\RBF\\Contest\\final\\output.k"
output_file = input_file.replace(".k", "_transformed.k")
file_lines = None
# 初始化函数：加载表面节点数据和目标坐标
def initialize_shell_data():
    global all_shell_points, node_id_to_transformed_coords, file_lines
    # 1. 获取表面节点和原始坐标
    nodes_method, nodes_param, range = "csv", "E:\\LZYZ\\Scoliosis\\RBF\\Contest\\final\\shell_property.csv", "A2:B35"
    all_shell_nodes = get_all_nodes(nodes_method, nodes_param, range)
    all_shell_points = np.array([[node._id] + list(node.position) for node in all_shell_nodes])

    # 2. 定义 shell_node_data 数组
    num_shell_nodes = len(all_shell_nodes)
    shell_node_data = np.empty((num_shell_nodes, 7), dtype=object)
    shell_node_data[:, 0] = [node._id for node in all_shell_nodes]
    shell_node_data[:, 1:4] = [node.position for node in all_shell_nodes]

    # 3. 加载目标坐标
    target_nodeids = set(int(point[0]) for point in all_shell_points)
    input_file = r"E:\\LZYZ\\Scoliosis\\RBF\\Contest\\final\\output.k"
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

def process_task(task_settings):
    """
    处理单个任务，计算变换后的节点并添加到全局的 all_transformed_points 列表中
    """
    # 获取当前任务的设置
    use_manual_source_nodes = task_settings.get("use_manual_source_nodes", False) #false是用xyz投影点
    transform_sets = task_settings["transform_sets"]  # 将集合转换为列表
    source_shell_nodes = task_settings["source_shell_nodes"]  # 假设已经是字符串列表
    total_side_num = task_settings.get("total_side_num", 0)  # 如果键不存在，返回 0
    total_plane_num = task_settings.get("total_plane_num", 0)  # 如果键不存在，返回 0
    modify_nodes = task_settings.get("modify_nodes", [])  # 用于修改特定节点的变换后的内部点
    node_direction_map = task_settings.get("node_direction_map", {})

    # 获取 transform_points
    transform_nodes = get_all_nodes("set", transform_sets)
    transform_points = np.array([[node._id] + list(node.position) for node in transform_nodes])

    # 获取控制点
    source_shell_nodes = get_all_nodes("pid", source_shell_nodes)
    if total_side_num == 0:
        print("SS`total_side_num` 和 `total_plane_num` 均为 0，直接使用 `source_shell_nodes` 作为 `source_select_nodes`")
        source_select_points = np.empty((0, 4))  # **确保是二维空数组**
        target_select_points = np.empty((0, 4))
    else:
        source_select_nodes = select_symmetric_uniform_nodes(source_shell_nodes, total_side_num, total_plane_num)
        source_select_points = np.array([[node._id] + list(node.position) for node in source_select_nodes])
        # 获取目标坐标
        target_select_points = get_target_coordinates(source_select_points)

    if use_manual_source_nodes:
        print("`use_manual_source_nodes=True`，不进行投影，仅使用 `source_shell_nodes` 进行 RBF 变换")
        print("source_select_points")
        print([int(point[0]) for point in source_select_points])
        print("target_select_points")
        print([int(point[0]) for point in target_select_points])
        # 进行 RBF 变换
        transformed_points = rbf_transform_3d_chunked(
            transform_points,
            source_select_points,
            target_select_points,
            0
        )
    else:
        print("`use_manual_source_nodes=False`，使用 XYZ 投影点计算 RBF 控制点")
        # 获取投影后的内部和外部定位点
        inner_locating_points, outer_locating_points, transformed_outer_locating_points, transformed_inner_locating_points = project_and_transform_to_surface(all_shell_points, node_direction_map)

        # **如果 `modify_nodes` 不为空，则进行修改**
        if modify_nodes:
            print("执行 modify_transformed_inner_points")
            transformed_inner_locating_points = modify_transformed_inner_points(transformed_inner_locating_points, modify_nodes)

        print("inner_locating_points")
        print([int(point[0]) for point in inner_locating_points])
        print("outer_locating_points")
        print([int(point[0]) for point in outer_locating_points])

        # 合并内部、外部和变换后的定位点
        # 使用 numpy 的 unique 方法去重

        # 合并所有 source points
        source_all_points = np.vstack([
            source_select_points,
            inner_locating_points,
            outer_locating_points
        ])
        print("source_all_points")
        # 直接打印所有的 node_id，格式为 int[node_id]
        print([int(point[0]) for point in source_all_points])
        # 合并所有 target points
        target_all_points = np.vstack([
            target_select_points,
            transformed_inner_locating_points,
            transformed_outer_locating_points
        ])
        print("target_all_points")
        print([int(point[0]) for point in target_all_points])
        # 使用 numpy.unique 来确保 node_id 不重复
        _, unique_source_idx = np.unique(source_all_points[:, 0], return_index=True)
        _, unique_target_idx = np.unique(target_all_points[:, 0], return_index=True)

        source_control_points = source_all_points[unique_source_idx]
        target_control_points = target_all_points[unique_target_idx]
        
        # 进行 RBF 变换
        transformed_points = rbf_transform_3d_chunked(
            transform_points,
            source_control_points,
            target_control_points,
            0
        )
    print(f"变换后的点数: {transformed_points.shape[0]}")

    # 将变换后的点添加到全局列表中
    all_transformed_points.append(transformed_points)


def write_all_transformed_points():
    """
    所有任务完成后，只写入一次所有的变换后的坐标
    """
    # 合并所有变换后的坐标
    all_transformed_points_combined = np.vstack(all_transformed_points)
    # 写入修改后的坐标
    write_modified_coordinates(
        output_file=output_file,
        file_lines=file_lines,
        updated_node_data=all_transformed_points_combined
    )
    print(f"所有任务完成，修改后的坐标已写入文件: {output_file}")

def run_batch(task_configs):
    """
    批量运行多个任务
    :param task_configs: 每个任务的配置（node_direction_map, transform_sets, etc.）
    :param file_lines: 输入文件的行数据
    """
    all_shell_points,node_id_to_transformed_coords, file_lines =initialize_shell_data()  # 加载表面节点数据和目标坐标
    
    # 遍历任务配置，处理每个任务
    for idx, task_settings in enumerate(task_configs):
        print(f"开始处理任务 {idx + 1}...")
        process_task(task_settings)

    # 在所有任务完成后写入合并后的变换结果
    write_all_transformed_points()


# 示例任务配置
task_configs = [
    {
        "node_direction_map": {
            'Xnegative': [(83010467, 'coccyx'),(89066294, 'Base of sacrum'),(89066279, 'Promontory'),(83012513,'ASIS-left'),(83512514,'ASIS-right'),
            (89004125,'Manubrium'), (89003192, 'xiphisternum'), (89059500, 'T1Am'),
            (89034241, 'Rib10LA'),(89034350, 'Rib8LA'), (89019613, 'Rib6LA'),(89018815,'Rib4LA'),(89018618,'Rib2LA'),(89018420,'Rib1LA'),
            (89534241, 'Rib10RA'),(89534350, 'Rib8RA'), (89519613, 'Rib6RA'),(89518815,'Rib4RA'),(89518618,'Rib2RA'),(89518420,'Rib1RA'),
            (88178168, 'Head C.G.'),(88178675, 'Glabella'),(88176493, 'Head Top'),(87000751,'C1Am'),
            (88265606,'eyeLU'),(88263462,'eyeRU'),(88175601,'eyeLL'),(88170878,'eyeRL'),(88261629,'nose'),
            (88143570,''),(88261630,''),(88135352,''),
            (88175279,''),(88170516,''),(88167682, 'chin'),
            (82008321,''),(82007994,''),(82008524,''),(82006458,''),(82008529,''),(82007910,'',),
            (81008321,''),(81007994,''),(81008524,''),(81006458,''),(81008529,''),(81007910,'',)
            ],
            'Xpositive': [(89063788, 'T12Pm'),(89000726,'T1Pm'),
            (89053454,'Rib1LP'),(89004188,'Rib3LP'),(89019330,'Rib5LP'),(89004321,'Rib7LP'),(89004461,'Rib9LP'),(89004320,'Rib11LP'),
            (89553454,'Rib1RP'),(89504188,'Rib3RP'),(89519330,'Rib5RP'),(89504321,'Rib7RP'),(89504461,'Rib9RP'),(89504320,'Rib11RP'),
            (87000262,'C1Pm'),
            (89006076,'Clavicle-Li'),(89009078,'Clavicle-Lo'),(89506076,'Clavicle-Ri'),(89509078,'Clavicle-Ro'),
            ],
            'Ynegative':[(88175941,'gonionL'),],
            'Ypositive':[(88171241,'gonionR')],
        },
        "transform_sets": ["101","102","103","104","105","106"],
        "source_shell_nodes": ["83200101", "83700101","89200801","89700801","88000222", "88000230","88000229", "88000231","88000221","88000223","87200101", "87700101","89200701", "89700701",
                               "82200001","82200401","82200601","82201101","82201301",
                               "81200001","81200401","81200601","81201101","81201301"
                               ],
        "total_side_num": 1000,
        "total_plane_num": 100,
        "modify_nodes": [(83010467, "X", 50),(89066294, "X", 50), (89066279, "X", 50),(83012513, "X", 50),(83512514, "X", 50),
                         (89063788, "Z", -20),(89063788, "X", -20),
                         (88261629, "X", -10),(88167682,"Z",-15),(88175941,"Z",-15),(88171241,"Z",-15),
                         (82008524,"Y",-4),(81008524,"Y",4),(82008529,"Y",-1),(81008529,"Y",1)
                         ]     
    },
]

# 执行批量任务
file_lines = []  # 这里应为文件行数据，可以通过 `read_node_coordinates` 获取
start_time = time.time()
run_batch(task_configs)

end_time = time.time()
print(f"文件名: {os.path.basename(__file__)}, 总运行时间: {end_time - start_time:.2f} s")