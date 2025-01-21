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


def get_nodes_from_pids(pids, deck=constants.LSDYNA):
    """
    根据单个或多个 PID 获取节点实体对象。

    参数：
        pids (str or list): 单个 PID（字符串）或多个 PID 的列表。
        deck: 求解器类型，默认为 LS-DYNA。

    返回：
        list: 包含节点实体对象的列表。
    """
    if isinstance(pids, str):  # 如果是单个 PID，转换为列表
        pids = [pids]

    unique_nodes = {}  # 用于去重的字典，key 为 node_id，value 为节点实体

    for pid in pids:
        # 获取 SECTION_SHELL
        section_shell = base.GetEntity(deck, "SECTION_SHELL", int(pid))
        # 获取 ELEMENT_SHELL
        elements = base.CollectEntities(deck, [section_shell], "ELEMENT_SHELL", recursive=True)
        # 提取节点
        for element in elements:
            node_values = element.get_entity_values(deck, ["N1", "N2", "N3", "N4"])
            for node in node_values.values():
                if isinstance(node, base.Entity):  # 确保是节点实体
                    if node._id not in unique_nodes:
                        unique_nodes[node._id] = node

    return list(unique_nodes.values())  # 返回节点实体对象列表


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

def get_all_nodes(method, param1, param2=None):
    """
    获取目标节点列表，支持三种方式：
    1. 从 CSV 文件读取 PID 列表。
    2. 直接传入 PID 列表。
    3. 通过 set_id 获取节点。
    
    参数：
        method (str): 获取节点的方式，"csv" / "pids" / "set"。
        param1 (str or list): 
            - 对于 "csv" 方法：CSV 文件路径。
            - 对于 "pids" 方法：PID 列表。
            - 对于 "set" 方法：set_id 列表。
        param2 (str or None): 可选参数，若 `method` 为 "csv"，则为读取的范围（如 "A2:B35"）；否则为 `None`。
    
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
        all_nodes = get_nodes_from_pids(pids)  # 获取对应 PIDs 的所有节点
        return all_nodes

    elif method == "pid":
        # 直接传入 PID 列表
        '''
        pids = ["1", "2"]
        # all_nodes = get_all_nodes("pid", pids)
        '''
        pids = param1
        all_nodes = get_nodes_from_pids(pids)  # 获取对应 PIDs 的所有节点
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





