import os
import sys
third_packages = r"G:\\anaconda3\\envs\\ansa_meta_env\\Lib\\site-packages"
sys.path.append(third_packages)

import ansa
from ansa import *
import numpy as np
import pandas as pd
from utils_node import *
from utils_adjust import *


def replace_set_coordinates_by_rule(set1_nodes, set2_nodes, reflected_coords, id_replace_rule):
    """
    根据映射规则替换节点坐标。

    参数：
    - set1_nodes: 源节点对象列表。
    - set2_nodes: 目标节点对象列表。
    - reflected_coords: NumPy 数组，对称后的节点坐标。
    - id_replace_rule: 函数，接收源节点 ID，返回目标节点 ID。

    返回：
    - 无直接返回值，直接修改目标节点的 `.position`。
    """
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
    """
    功能：
        使用映射文件将源节点的对称坐标替换到目标节点。

    输入：
        - set1_nodes: 源节点对象列表，每个节点具有 ._id 和 .position 属性；
        - set2_nodes: 目标节点对象列表，每个节点具有 ._id 和 .position 属性；
        - reflected_coords: NumPy 数组，形状为 (N, 3)，表示源节点的对称坐标；
        - mapping_file: 字符串，表示映射文件路径，文件中需要包含 "Left_Node_ID" 和 "Right_Node_ID" 列。

    输出：
        - 无直接返回值，直接修改目标节点对象的 .position 属性。
    """
    try:
        df = pd.read_csv(mapping_file)
        if "Left_Node_ID" not in df.columns or "Right_Node_ID" not in df.columns:
            raise ValueError("映射文件缺少必要的列：Left_Node_ID 或 Right_Node_ID")
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
    """
    # 获取节点
    source_set_nodes = get_all_nodes(source_set_method, source_set_identifiers)
    target_set_nodes = get_all_nodes(target_set_method, target_set_identifiers)
    set56_nodes = get_all_nodes(set56_method, set56_identifiers)

    # 计算反射平面
    normal, point_on_plane = calculate_dynamic_reflection_plane(set56_nodes)

    # 提取坐标并打印
    source_set_coords = np.array([node.position for node in source_set_nodes])

    # 反射变换
    reflected_coords = reflect_coordinates_with_near_plane_handling(source_set_coords, normal, point_on_plane)

    # 替换逻辑
    if task_type == "rule":
        replace_set_coordinates_by_rule(source_set_nodes, target_set_nodes, reflected_coords, id_replace_rule)
    elif task_type == "mapping":
        replace_set_coordinates_by_mapping(source_set_nodes, target_set_nodes, reflected_coords, mapping_file)
    else:
        print("未知任务类型！")

    print("对称替换完成！")


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
    set56_identifiers = ["3","4","5","6"]  # 对称平面的 Set ID

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
        # 基于映射文件的替换  头
        {
            "task_type": "mapping",
            "source_set": {"method": "set", "identifiers": ["41"]},
            "target_set": {"method": "set", "identifiers": ["42"]},
            "mapping_file": "node_mapping.csv"
        },       

        # 替换规则 870 -> 875 颈
        {
            "task_type": "rule",
            "source_set": {"method": "set", "identifiers": [47]},
            "target_set": {"method": "set", "identifiers": [48]},
            "id_replace_rule": lambda x: int(str(x).replace("870", "875", 1))
        },
        # 替换规则 890 -> 895 胸
        {
            "task_type": "rule",
            "source_set": {"method": "set", "identifiers": [43]},
            "target_set": {"method": "set", "identifiers": [44]},
            "id_replace_rule": lambda x: int(str(x).replace("890", "895", 1))
        },
        # 替换规则 830 -> 835 臀
        {
            "task_type": "rule",
            "source_set": {"method": "set", "identifiers": [45]},
            "target_set": {"method": "set", "identifiers": [46]},
            "id_replace_rule": lambda x: int(str(x).replace("830", "835", 1))
        },
        # 替换规则 82 -> 81 腿
        {
            "task_type": "rule",
            "source_set": {"method": "set", "identifiers": [49]},
            "target_set": {"method": "set", "identifiers": [40]},
            "id_replace_rule": lambda x: int(str(x).replace("82", "81", 1))
        },
    ]

    # 执行批量任务
    batch_run(runs, set56_method, set56_identifiers)




