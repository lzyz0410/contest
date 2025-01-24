import os
import sys
third_packages = r"G:\\anaconda3\\envs\\ansa_meta_env\\Lib\\site-packages"
sys.path.append(third_packages)

import ansa
from ansa import *
import numpy as np
import time
import pandas as pd
from utils_node import *
from utils_rbf_transform import *
from utils_smooth import *
from utils_reflect import reflect

from scipy.spatial import KDTree
import numpy as np
from scipy.interpolate import splprep, splev

def sort_nodes_by_proximity(nodes):
    """
    按节点的空间距离排序，确保顺序符合几何分布。
    
    参数：
    - nodes: 节点对象列表，包含 .position 属性 (tuple: (x, y, z))。
    
    返回：
    - 排序后的节点列表。
    """
    if not nodes:
        return []

    sorted_nodes = [nodes[0]]  # 从第一个节点开始
    remaining_nodes = nodes[1:]

    while remaining_nodes:
        # 找到当前节点最近的节点
        current_node = sorted_nodes[-1]
        current_pos = np.array(current_node.position)
        distances = [np.linalg.norm(np.array(node.position) - current_pos) for node in remaining_nodes]
        nearest_idx = np.argmin(distances)
        # 将最近的节点加入排序列表
        sorted_nodes.append(remaining_nodes.pop(nearest_idx))

    return sorted_nodes


def smooth_set_nodes(set_nodes, smoothing_factor=0.1, ensure_closure=True):
    """
    对 set_nodes 的坐标进行平滑处理，并确保首尾光滑闭合（可选）。

    参数：
    - set_nodes: 节点对象列表，包含 .position 属性 (tuple: (x, y, z))。
    - smoothing_factor: 平滑因子，用于控制拟合曲线的平滑程度（默认值为 0.1）。
    - ensure_closure: 是否确保首尾光滑闭合（默认值为 True）。

    返回：
    - 无直接返回值，直接修改 set_nodes 的 .position 属性为平滑后的坐标。
    """
    if not set_nodes:
        print("节点列表为空，无法执行平滑操作。")
        return

    # 按空间顺序对节点排序
    print("对节点按空间距离进行排序...")
    sorted_nodes = sort_nodes_by_proximity(set_nodes)

    # 提取排序后的坐标
    coords = np.array([node.position for node in sorted_nodes])
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # 如果需要闭合，则在样条拟合中指定周期性
    print("开始拟合 set_nodes 的平滑曲线...")
    tck, u = splprep([x, y, z], s=len(sorted_nodes) * smoothing_factor, k=3, per=ensure_closure)

    # 生成平滑曲线上的点
    fitted_coords = splev(np.linspace(0, 1, len(sorted_nodes)), tck)

    # 更新 set_nodes 的位置为平滑曲线上的点
    print("更新 set_nodes 的位置为拟合曲线点...")
    for i, node in enumerate(sorted_nodes):
        node.position = tuple(fitted_coords[j][i] for j in range(3))
        print(f"点 ID: {node._id}, 更新位置为: {node.position}")

# 示例调用
set_nodes = get_all_nodes("set", ["20"])
smooth_set_nodes(set_nodes, smoothing_factor=10, ensure_closure=True)
