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
from utils_smooth import *

def force_unify_boundary_axis(boundary_nodes, axis):
    """
    强制将分界点的指定轴坐标统一为平均值。
    参数：
    - boundary_nodes: 分界点节点对象列表。
    - axis: 指定轴 (1 表示 Y，2 表示 Z)。
    返回：
    - average_value: 分界点统一后的指定轴均值。
    """
    boundary_coords = np.array([node.position for node in boundary_nodes])
    average_value = np.mean(boundary_coords[:, axis])
    print(f"强制分界点 {['X', 'Y', 'Z'][axis]} 坐标统一为 {average_value:.4f}")
    for node in boundary_nodes:
        pos = list(node.position)
        pos[axis] = average_value
        node.position = tuple(pos)
    return average_value


def smooth_nearby_nodes(nearby_nodes, boundary_value, axis, smoothing_factor=0.5, condition="above"):
    """
    平滑调整点集，使其满足指定的条件（高于/低于分界点）。
    参数：
    - nearby_nodes: 要调整的点节点列表。
    - boundary_value: 分界点的统一值。
    - axis: 处理的坐标轴（1 表示 Y，2 表示 Z）。
    - smoothing_factor: 平滑强度，范围 (0, 1]。
    - condition: 指定调整条件（"above" 表示高于，"below" 表示低于）。
    """
    for node in nearby_nodes:
        pos = list(node.position)
        if (condition == "above" and pos[axis] > boundary_value) or \
           (condition == "below" and pos[axis] < boundary_value):
            pos[axis] = pos[axis] * (1 - smoothing_factor) + (boundary_value - 1e-3 if condition == "above" else boundary_value + 1e-3) * smoothing_factor
            node.position = tuple(pos)


def get_plane_y_value(set_nodes):
    """
    从平面节点集合中获取固定的 Y 值。
    参数：
    - set_nodes: 用于定义平面的节点对象列表。
    返回：
    - y_value: 平面上的固定 Y 值。
    """
    coords = np.array([node.position for node in set_nodes])
    if len(coords) < 1:
        raise ValueError("定义平面需要至少 1 个点！")
    y_value = np.mean(coords[:, 1])
    print(f"平面确定的固定 Y 值: {y_value}")
    return y_value


def align_boundary_to_y_value(boundary_nodes, y_value):
    """
    调整分界点，使其 Y 坐标等于平面固定 Y 值。
    参数：
    - boundary_nodes: 分界点的节点对象列表。
    - y_value: 平面确定的固定 Y 值。
    """
    for node in boundary_nodes:
        pos = list(node.position)
        pos[1] = y_value  # 强制设置 Y 坐标
        node.position = tuple(pos)
    print(f"分界点 Y 坐标已统一为平面 Y 值 {y_value:.4f}！")


def smooth_nearby_nodes_y(nearby_nodes, y_value, smoothing_factor=0.5):
    """
    平滑调整 `nearby_set` 中的点，使其 Y 坐标小于平面固定 Y 值。
    参数：
    - nearby_nodes: 要调整的点节点列表。
    - y_value: 平面确定的固定 Y 值。
    - smoothing_factor: 平滑强度，范围 (0, 1]。
    """
    for node in nearby_nodes:
        pos = list(node.position)
        if pos[1] >= y_value:
            pos[1] = pos[1] * (1 - smoothing_factor) + (y_value - 1e-3) * smoothing_factor
            node.position = tuple(pos)
    print("周围点 Y 坐标已平滑调整！")


def main(task_type,plane_set=None, axis=1, smoothing_factor=0.9):
    """
    主函数：根据任务类型完成调整。
    参数：
        task_type: 任务类型（"axis" 或 "plane"）。
        boundary_set: 分界点的 Set ID。
        nearby_set: 周围点的 Set ID。
        plane_set: 用于定义平面的 Set ID，仅对 "plane" 类型有效。
        axis: 指定强制对齐的轴，仅对 "axis" 类型有效（1 表示 Y，2 表示 Z）。
        smoothing_factor: 平滑强度，范围 (0, 1]。
    """
    print(f"运行任务类型: {task_type}")
    
    # 获取分界点和周围点
    boundary_set = [3]
    boundary_nodes = get_nodes_from_set(boundary_set)
    #nearby_nodes = get_all_nodes("pid",["88000229","88000230","88000231"])
    nearby_nodes = get_all_nodes("pid",["88000221","88000222","88000223"])

    if task_type == "plane":
        if not plane_set:
            raise ValueError("任务类型 'plane' 必须提供 plane_set！")
        print(f"平面定义的 Set ID: {plane_set}")
        plane_nodes = get_nodes_from_set([plane_set])
        
        # 计算平面的 Y 值
        y_value = get_plane_y_value(plane_nodes)
        
        # 调整分界点和周围点
        align_boundary_to_y_value(boundary_nodes, y_value)
        smooth_nearby_nodes_y(nearby_nodes, y_value, smoothing_factor)

    elif task_type == "axis":
        if axis not in [1, 2]:
            raise ValueError("任务类型 'axis' 必须指定轴（1 表示 Y，2 表示 Z）！")
        print(f"指定调整的轴: {['X', 'Y', 'Z'][axis]}")
        
        # 强制统一分界点的轴坐标
        boundary_value = force_unify_boundary_axis(boundary_nodes, axis)
        
        # 平滑调整周围点
        condition = "above" if axis == 2 else "below"
        smooth_nearby_nodes(nearby_nodes, boundary_value, axis, smoothing_factor, condition)

    else:
        raise ValueError(f"未知任务类型: {task_type}")

    print(f"任务 {task_type} 已完成！")

    # 定义批量运行的任务
    # runs = [
    #     {"task_type": "axis", "boundary_set": 51, "nearby_set": 31, "axis": 2, "smoothing_factor": 0.9},
    #     {"task_type": "axis", "boundary_set": 52, "nearby_set": 32, "axis": 2, "smoothing_factor": 0.9},
    #     {"task_type": "axis", "boundary_set": 53, "nearby_set": 33, "axis": 2, "smoothing_factor": 0.9},
    #     {"task_type": "axis", "boundary_set": 54, "nearby_set": 34, "axis": 2, "smoothing_factor": 0.9},
    #     {"task_type": "axis", "boundary_set": 55, "nearby_set": 35, "axis": 2, "smoothing_factor": 0.9},
    #     {"task_type": "axis", "boundary_set": 56, "nearby_set": 35, "axis": 1, "smoothing_factor": 0.9},
    #     {"task_type": "plane", "boundary_set": 57, "nearby_set": 66, "plane_set": 56, "smoothing_factor": 0.9},
    #     {"task_type": "plane", "boundary_set": 58, "nearby_set": 67, "plane_set": 56, "smoothing_factor": 0.9},
    #     {"task_type": "plane", "boundary_set": 59, "nearby_set": 68, "plane_set": 56, "smoothing_factor": 0.9},
    # ]
