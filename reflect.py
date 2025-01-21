import os
import sys
third_packages = r"G:\\anaconda3\\envs\\ansa_meta_env\\Lib\\site-packages"
sys.path.append(third_packages)

import ansa
from ansa import *
import numpy as np
import pandas as pd
from utils_node import *

# 计算反射平面
def calculate_reflection_plane(set_nodes):
    coords = np.array([node.position for node in set_nodes])
    if len(coords) < 3:
        raise ValueError("定义平面需要至少 3 个点！")
    p1, p2, p3 = coords[:3]
    normal = np.cross(p2 - p1, p3 - p1)  # 计算法向量
    normal = normal / np.linalg.norm(normal)  # 单位化法向量
    point_on_plane = p1  # 平面上的一个点
    print(f"平面法向量: {normal}, 平面上的点: {point_on_plane}")
    return normal, point_on_plane

# 对称变换
def reflect_coordinates(coords, normal, point_on_plane):
    reflected_coords = []
    for coord in coords:
        vector_to_plane = coord - point_on_plane
        distance_to_plane = np.dot(vector_to_plane, normal)
        reflected_coord = coord - 2 * distance_to_plane * normal  # 对称变换公式
        reflected_coords.append(reflected_coord)
    return np.array(reflected_coords)

# 替换坐标（基于替换规则）
def replace_set_coordinates_by_rule(set1_nodes, set2_nodes, reflected_coords, id_replace_rule):
    set2_id_map = {node._id: node for node in set2_nodes}
    for node, new_coord in zip(set1_nodes, reflected_coords):
        set1_id = node._id
        set2_id = id_replace_rule(set1_id)
        if set2_id in set2_id_map:
            set2_node = set2_id_map[set2_id]
            set2_node.position = tuple(new_coord)
        else:
            print(f"Set2 中未找到与 Set1 节点 ID {set1_id} 对应的节点 ID {set2_id}！")

# 替换坐标（基于映射文件）
def replace_set_coordinates_by_mapping(set1_nodes, set2_nodes, reflected_coords, mapping_file):
    try:
        df = pd.read_csv(mapping_file)
        mapping = dict(zip(df["Left_Node_ID"], df["Right_Node_ID"]))
        print(f"从文件 {mapping_file} 成功加载节点映射！")
    except Exception as e:
        print(f"加载映射文件失败: {e}")
        return

    set2_id_map = {node._id: node for node in set2_nodes}
    for node, new_coord in zip(set1_nodes, reflected_coords):
        set1_id = node._id
        set2_id = mapping.get(set1_id)
        if set2_id in set2_id_map:
            set2_node = set2_id_map[set2_id]
            set2_node.position = tuple(new_coord)
        else:
            print(f"Set2 中未找到与 Set1 节点 ID {set1_id} 对应的节点 ID {set2_id}！")

# 主函数
def main(task_type, source_set_method, source_set_identifiers, target_set_method, target_set_identifiers, set56_method, set56_identifiers, mapping_file=None, id_replace_rule=None):
    """
    主函数：处理替换任务。
    参数：
    - task_type: 任务类型（"rule" 或 "mapping"）。
    - source_set_method: 获取 source set 节点的方式，"pid" 或 "set"。
    - source_set_identifiers: source set 的标识符列表（pid 或 set id）。
    - target_set_method: 获取 target set 节点的方式，"pid" 或 "set"。
    - target_set_identifiers: target set 的标识符列表（pid 或 set id）。
    - set56_method: 获取对称平面 set 节点的方式，"pid" 或 "set"。
    - set56_identifiers: 对称平面 set 的标识符列表（pid 或 set id）。
    - mapping_file: 节点映射表文件路径（适用于 "mapping" 类型）。
    - id_replace_rule: 替换规则函数（适用于 "rule" 类型）。
    """
    # 获取 source set、target set 和对称平面 set 的节点
    source_set_nodes = get_all_nodes(source_set_method, source_set_identifiers)
    target_set_nodes = get_all_nodes(target_set_method, target_set_identifiers)
    set56_nodes = get_all_nodes(set56_method, set56_identifiers)

    # 计算反射平面
    normal, point_on_plane = calculate_reflection_plane(set56_nodes)

    # 提取 source set 的坐标
    source_set_coords = np.array([node.position for node in source_set_nodes])

    # 对称变换
    reflected_coords = reflect_coordinates(source_set_coords, normal, point_on_plane)

    # 根据任务类型执行替换
    if task_type == "rule":
        replace_set_coordinates_by_rule(source_set_nodes, target_set_nodes, reflected_coords, id_replace_rule)
    elif task_type == "mapping":
        replace_set_coordinates_by_mapping(source_set_nodes, target_set_nodes, reflected_coords, mapping_file)
    else:
        print("未知任务类型！")

    print(f"Source set 的对称坐标已成功替换到 Target set！")


# 批量任务运行
def batch_run(runs, set56_method, set56_identifiers):
    for i, task in enumerate(runs, start=1):
        print(f"\n批量运行第 {i} 个任务：")
        source_set = task["source_set"]
        target_set = task["target_set"]
        main(task["task_type"], source_set["method"], source_set["identifiers"], target_set["method"], target_set["identifiers"], set56_method, set56_identifiers, 
             task.get("mapping_file"), task.get("id_replace_rule"))


if __name__ == "__main__":
    set56_method = "set"  # 对称平面的 set 获取方式
    set56_identifiers = [3]  # 对称平面的 Set ID

    # 定义批量任务
    runs = [
        # 替换规则 86 -> 85 手臂
        {
            "task_type": "rule",
            "source_set": {"method": "pid", "identifiers": ["86200001", "86200301", "86200501", "86200801", "86201001"]},
            "target_set": {"method": "pid", "identifiers": ["85200001", "85200301", "85200501", "85200801", "85201001"]},
            "id_replace_rule": lambda x: int(str(x).replace("86", "85", 1))
        },

        # 替换规则 肩膀
        {
            "task_type": "rule",
            "source_set": {"method": "pid", "identifiers": ["89200701"]},
            "target_set": {"method": "pid", "identifiers": ["89700701"]},
            "id_replace_rule": lambda x: x + 500000  # 替换规则：target 的 ID = source 的 ID + 500,000
        },
        
        # # 替换规则 835 -> 830
        # {
        #     "task_type": "rule",
        #     "source_set": {"method": "set", "identifiers": [60]},
        #     "target_set": {"method": "set", "identifiers": [61]},
        #     "id_replace_rule": lambda x: int(str(x).replace("835", "830", 1))
        # },

        # # 基于映射文件的替换  头
        # {
        #     "task_type": "mapping",
        #     "source_set": {"method": "set", "identifiers": ["41"]},
        #     "target_set": {"method": "set", "identifiers": ["42"]},
        #     "mapping_file": "node_mapping.csv"
        # }
    ]

    # 执行批量任务
    batch_run(runs, set56_method, set56_identifiers)




