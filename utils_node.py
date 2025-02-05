import os
import time
import pandas as pd
import numpy as np
import re
import sys

# 添加 ANSA 环境
third_packages = r"G:\\anaconda3\\envs\\ansa_meta_env\\Lib\\site-packages"
sys.path.append(third_packages)

import ansa
from ansa import *
from utils_data import read_csv_by_two_columns

def get_nodes_from_set(set_id):
    """
    根据 set_id 获取 ANSA 中的节点对象列表。
    
    参数：
        set_id (list or int or str): set 的 ID，可以是单个 ID 或多个 ID 的列表。
    
    返回：
        list: 节点对象的列表。
    """
    # 如果 set_id 是字符串或字符串列表，转换为整数
    if isinstance(set_id, str):
        set_id = int(set_id)
    elif isinstance(set_id, list):
        set_id = [int(sid) for sid in set_id]

    nodes = []  # 用于存储所有节点对象

    # 如果是列表，遍历每个 set_id 获取节点；否则处理单个 set_id
    if isinstance(set_id, list):
        for sid in set_id:
            set_entity = base.GetEntity(constants.LSDYNA, 'SET', sid)
            if set_entity:
                nodes.extend(base.CollectEntities(constants.LSDYNA, set_entity, 'NODE'))
    else:
        set_entity = base.GetEntity(constants.LSDYNA, 'SET', set_id)
        if set_entity:
            nodes.extend(base.CollectEntities(constants.LSDYNA, set_entity, 'NODE'))

    return nodes


def get_nodes_from_ids(node_ids):
    """
    根据节点 ID 获取节点实体对象。

    参数：
        node_ids (list): 节点 ID 列表。

    返回：
        list: 包含节点实体对象的列表。
    """
    node_ids_int = [int(node_id) for node_id in node_ids]  # 批量转换为整数
    nodes = []
    for node_id in node_ids_int:
        node = base.GetEntity(constants.LSDYNA, 'NODE', node_id)  # 获取节点实体
        if node:
            nodes.append(node)
        else:
            print(f"无法找到节点 ID: {node_id}")
    return nodes  # 返回节点实体对象列表


# def get_nodes_from_pids(pids, deck=constants.LSDYNA, filter_by_prefix=False):
#     """
#     根据单个或多个 PID 获取节点实体对象。
#     如果 filter_by_prefix 为 True 且节点 ID 的前两位不唯一，则保留数量最多的前缀对应的节点。

#     参数：
#         pids (str or list): 单个 PID（字符串）或多个 PID 的列表。
#         deck: 求解器类型，默认为 LS-DYNA。
#         filter_by_prefix (bool): 是否根据节点 ID 的前两位进行过滤，默认为 False。

#     返回：
#         list: 包含节点实体对象的列表。
#     """
#     if isinstance(pids, str):  # 如果是单个 PID，转换为列表
#         pids = [pids]

#     unique_nodes = {}  # 用于去重的字典，key 为 node_id，value 为节点实体
#     prefix_count = {}  # 统计每种前缀的节点数量

#     for pid in pids:
#         # 获取 SECTION_SHELL
#         section_shell = base.GetEntity(deck, "SECTION_SHELL", int(pid))
#         # 获取 ELEMENT_SHELL
#         elements = base.CollectEntities(deck, [section_shell], "ELEMENT_SHELL", recursive=True)
#         # 提取节点
#         for element in elements:
#             node_values = element.get_entity_values(deck, ["N1", "N2", "N3", "N4"])
#             for node in node_values.values():
#                 if isinstance(node, base.Entity):  # 确保是节点实体
#                     if node._id not in unique_nodes:
#                         unique_nodes[node._id] = node
#                         # 统计前缀
#                         prefix = str(node._id)[:2]  # 取前两位作为前缀
#                         if prefix in prefix_count:
#                             prefix_count[prefix] += 1
#                         else:
#                             prefix_count[prefix] = 1

#     # 如果不需要根据前缀过滤，直接返回所有节点
#     if not filter_by_prefix:
#         print("未启用前缀过滤，返回所有节点")
#         return list(unique_nodes.values())

#     # 判断前缀是否唯一
#     if len(prefix_count) == 1:
#         # 如果前缀唯一，直接返回所有节点
#         print("所有节点的前缀相同，无需过滤")
#         return list(unique_nodes.values())
#     else:
#         # 找到最大的数量
#         max_count = max(prefix_count.values())
#         # 找出所有数量等于最大数量的前缀
#         max_prefixes = [prefix for prefix, count in prefix_count.items() if count == max_count]

#         if len(max_prefixes) == 1:
#             max_prefix = max_prefixes[0]
#             print(f"前缀不唯一，保留最多的前缀: {max_prefix}, 数量: {prefix_count[max_prefix]}")
#         else:
#             print(f"有多个前缀数量相等且最多: {max_prefixes}, 数量: {max_count}，都将保留")

#         # 过滤节点，只保留最多前缀的节点
#         filtered_nodes = [node for node in unique_nodes.values() if str(node._id).startswith(tuple(max_prefixes))]
#         # 提取过滤后的节点 ID 并打印成列表形式
#         filtered_node_ids = [node._id for node in filtered_nodes]
#         #print("过滤后的节点 ID 列表：", filtered_node_ids)
#         return filtered_nodes


def get_nodes_from_pids(pids, deck=constants.LSDYNA, filter_by_prefix=False):
    """
    根据单个或多个 PID 获取节点实体对象，同时支持 ELEMENT_SHELL 和 ELEMENT_SOLID。
    如果 filter_by_prefix 为 True 且节点 ID 的前两位不唯一，则保留数量最多的前缀对应的节点。

    参数：
        pids (str or list): 单个 PID（字符串）或多个 PID 的列表。
        deck: 求解器类型，默认为 LS-DYNA。
        filter_by_prefix (bool): 是否根据节点 ID 的前两位进行过滤，默认为 False。

    返回：
        list: 包含节点实体对象的列表。
    """
    if isinstance(pids, str):  # 如果是单个 PID，转换为列表
        pids = [pids]

    unique_nodes = {}  # 用于去重的字典，key 为 node_id，value 为节点实体
    prefix_count = {}  # 统计每种前缀的节点数量

    for pid in pids:
        # 获取 SECTION_SHELL 或 SECTION_SOLID
        section_shell = base.GetEntity(deck, "SECTION_SHELL", int(pid))
        section_solid = base.GetEntity(deck, "SECTION_SOLID", int(pid))

        # 获取 ELEMENT_SHELL
        elements_shell = base.CollectEntities(deck, [section_shell], "ELEMENT_SHELL", recursive=True) if section_shell else []
        # 获取 ELEMENT_SOLID
        elements_solid = base.CollectEntities(deck, [section_solid], "ELEMENT_SOLID", recursive=True) if section_solid else []

        # 处理 ELEMENT_SHELL
        for element in elements_shell:
            node_values = element.get_entity_values(deck, ["N1", "N2", "N3", "N4"])
            for node in node_values.values():
                if isinstance(node, base.Entity):  # 确保是节点实体
                    if node._id not in unique_nodes:
                        unique_nodes[node._id] = node
                        prefix = str(node._id)[:2]  # 取前两位作为前缀
                        prefix_count[prefix] = prefix_count.get(prefix, 0) + 1

        # 处理 ELEMENT_SOLID
        for element in elements_solid:
            node_values = element.get_entity_values(deck, ["N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8"])
            for node in node_values.values():
                if isinstance(node, base.Entity):  # 确保是节点实体
                    if node._id not in unique_nodes:
                        unique_nodes[node._id] = node
                        prefix = str(node._id)[:2]  # 取前两位作为前缀
                        prefix_count[prefix] = prefix_count.get(prefix, 0) + 1

    # 如果不需要根据前缀过滤，直接返回所有节点
    if not filter_by_prefix:
        print("未启用前缀过滤，返回所有节点")
        return list(unique_nodes.values())

    # 判断前缀是否唯一
    if len(prefix_count) == 1:
        print("所有节点的前缀相同，无需过滤")
        return list(unique_nodes.values())

    # 找到出现最多的前缀
    max_count = max(prefix_count.values())
    max_prefixes = [prefix for prefix, count in prefix_count.items() if count == max_count]

    if len(max_prefixes) == 1:
        max_prefix = max_prefixes[0]
        print(f"前缀不唯一，保留最多的前缀: {max_prefix}, 数量: {prefix_count[max_prefix]}")
    else:
        print(f"有多个前缀数量相等且最多: {max_prefixes}, 数量: {max_count}，都将保留")

    # 过滤节点，只保留出现次数最多的前缀
    filtered_nodes = [node for node in unique_nodes.values() if str(node._id).startswith(tuple(max_prefixes))]
    return filtered_nodes

def update_ansa_node_coordinates(transformed_points, all_nodes):
    """
    更新 ANSA 中的节点坐标。

    参数：
        transformed_points (numpy array): 变换后的节点坐标（形状: (N, 4)）。
        nodes (list): 节点实体对象的列表。
    """
    transformed_coords = transformed_points[:, 1:]  # (N x 3)
    # 遍历节点和对应的坐标
    for node, coords in zip(all_nodes, transformed_coords):
        # 确保 coords 是一个 3 元组
        coords_tuple = tuple(coords[:3])  # 只取前 3 个坐标，并转换为元组
        node.position = coords_tuple  # 更新节点坐标

def get_all_nodes(method, param1, param2=None, filter_by_prefix=False):
    """
    获取目标节点列表，支持三种方式：
    1. 从 CSV 文件读取 PID 列表。
    2. 直接传入 PID 列表。
    3. 通过 set_id 获取节点。
    4。通过nodeid获取节点。

    参数：
        method (str): 获取节点的方式，"csv" / "pids" / "set"/"node"。
        param1 (str or list): 
            - 对于 "csv" 方法：CSV 文件路径。
            - 对于 "pids" 方法：PID 列表。
            - 对于 "set" 方法：set_id 列表。
            - 对于 "node"方法：nodeid列表。
        param2 (str or None): 可选参数，若 `method` 为 "csv"，则为读取的范围（如 "A2:B35"）；否则为 `None`。
        filter_by_prefix (bool): 是否根据节点 ID 的前两位进行过滤，默认为 False。

    返回：
        list: 对应获取方式的目标节点对象列表。
    """
    if method == "csv":
        # 从 CSV 文件读取 PID 列表
        '''
        shell_property = "E:\\LZYZ\\Scoliosis\\RBF\\Contest\\final\\shell_property.csv"
        all_nodes = get_all_nodes("csv", shell_property, "A2:B35")
        '''
        csv_path = param1
        range_str = param2
        pid_name_pairs = read_csv_by_two_columns(csv_path, range_str)
        pids = [str(pid).strip() for _, pid in pid_name_pairs]
        #pid_names = {str(pid).strip(): name for name, pid in pid_name_pairs}
        all_nodes = get_nodes_from_pids(pids, filter_by_prefix=filter_by_prefix)  # 获取对应 PIDs 的所有节点
        return all_nodes

    elif method == "pid":
        # 直接传入 PID 列表
        '''
        pids = ["1", "2"]
        # all_nodes = get_all_nodes("pid", pids)
        '''
        pids = param1
        all_nodes = get_nodes_from_pids(pids, filter_by_prefix=filter_by_prefix)  # 获取对应 PIDs 的所有节点
        return all_nodes

    elif method == "set":
        # 通过 set_id 获取节点
        '''
            # set_ids = ["19","20"]
            # all_nodes = get_all_nodes("set", set_ids)
        '''
        set_ids = param1
        all_nodes = get_nodes_from_set(set_ids)  # 通过 set_id 获取对应的所有节点
        return all_nodes

    elif method == "node":
        # 通过nodeid获取节点  node_ids = ["100","300"]
        node_ids = param1
        all_nodes = get_nodes_from_ids(node_ids)  # 通过nodeid获取对应的所有节点
        return all_nodes





