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
from utils_smooth import *
from scipy.spatial import cKDTree
from utils_reflect import reflect

# 计算点集的几何中心
def calculate_geometric_center(nodes):
    """
    计算一组三维点的几何中心（质心）。
    
    参数:
        nodes (list): 节点列表，每个节点对象包含 position 属性。
    
    返回:
        np.array: 计算得到的几何中心坐标 (x_c, y_c, z_c)。
    """
    # 提取节点坐标
    coords = np.array([node.position for node in nodes])
    center = np.mean(coords, axis=0)  # 计算几何中心
    print(f"几何中心: {center}")
    return center

# 优化后的节点旋转函数，使用一次性矩阵运算处理
def rotate_nodes_from_set_optimized(all_points, center, rotation_axis, angle_degrees):
    """
    批量旋转节点并更新。
    参数:
        all_points (np.array): 包含节点 ID 和坐标的数组，形状为 (N, 4)，每行 [node_id, x, y, z]。
        center (np.array): 旋转中心。
        rotation_axis (list): 旋转轴（例如 [1, 0, 0] 表示绕 X 轴旋转）。
        angle_degrees (float): 旋转角度，以度为单位。
    
    返回:
        np.array: 包含节点 ID 和旋转后坐标的数组，每行 [node_id, x, y, z]。
    """
    # 构建旋转矩阵
    ux, uy, uz = rotation_axis
    theta = np.radians(angle_degrees)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    one_minus_cos = 1 - cos_theta

    # 旋转矩阵
    R = np.array([
        [cos_theta + ux**2 * one_minus_cos,
         ux * uy * one_minus_cos - uz * sin_theta,
         ux * uz * one_minus_cos + uy * sin_theta],
        
        [uy * ux * one_minus_cos + uz * sin_theta,
         cos_theta + uy**2 * one_minus_cos,
         uy * uz * one_minus_cos - ux * sin_theta],
        
        [uz * ux * one_minus_cos - uy * sin_theta,
         uz * uy * one_minus_cos + ux * sin_theta,
         cos_theta + uz**2 * one_minus_cos]
    ])
    
    # 提取节点 ID 和节点坐标
    node_ids = all_points[:, 0]  # 第一列是节点 ID
    node_positions = all_points[:, 1:]  # 后三列是节点坐标 [x, y, z]

    # 旋转所有节点的坐标
    rotated_positions = (node_positions - center) @ R.T + center

    # 将节点 ID 和旋转后的坐标合并
    rotated_all_points = np.column_stack((node_ids, rotated_positions))

    return rotated_all_points

def get_control_points4(all_points, rotated_all_points, control_fixed_method, control_fixed_param, 
                        control_moving_method, control_moving_param, 
                        uniform_fixed_node_count=None, uniform_moving_node_count=None):
    """
    获取控制点（固定点和运动点），并返回它们的源和目标坐标。
    
    参数:
        all_points (np.array): 所有节点的原始坐标，形状为 (N, 4)，每行包含 [node_id, x, y, z]。
        rotated_all_points (np.array): 所有节点的旋转后坐标，形状为 (N, 4)，每行包含 [node_id, x, y, z]。
        fixed_method (str): 获取固定控制点的方式，"pid" 或 "set"。
        fixed_param (list): 获取固定控制点的参数（PID 或 set_id 列表）。
        moving_method (str): 获取运动控制点的方式，"pid" 或 "set"。
        moving_param (list): 获取运动控制点的参数（PID 或 set_id 列表）。
        uniform_fixed_node_count (int, optional): 如果给定，将使用 `select_uniform_nodes` 来减少固定控制点的数量。默认为 `None`，表示不做筛选。
        uniform_moving_node_count (int, optional): 如果给定，将使用 `select_uniform_nodes` 来减少运动控制点的数量。默认为 `None`，表示不做筛选。
    
    
    返回:
        source_control_points (np.array): 固定点和运动点的源坐标。
        target_control_points (np.array): 固定点和运动点的目标坐标。
    """
    # 获取固定控制点
    fixed_control_nodes = get_all_nodes(control_fixed_method, control_fixed_param)
    fixed_points = np.array([[node._id] + list(node.position) for node in fixed_control_nodes])

    # 如果指定了均匀节点数量，应用 select_uniform_nodes 来选择固定控制点
    if uniform_fixed_node_count is not None:
        fixed_control_nodes = select_uniform_nodes(fixed_control_nodes, uniform_fixed_node_count)
        fixed_points = np.array([[node._id] + list(node.position) for node in fixed_control_nodes])
    #print(f"固定控制点ID: {[node._id for node in fixed_control_nodes]}")

    # 获取运动控制点
    moving_control_nodes = get_all_nodes(control_moving_method, control_moving_param)

    # 获取运动控制点的 source 和 target 坐标
    moving_source_points = np.array([[node._id] + list(all_points[all_points[:, 0] == node._id][0, 1:]) 
                                     for node in moving_control_nodes])
    moving_target_points = np.array([[node._id] + list(rotated_all_points[rotated_all_points[:, 0] == node._id][0, 1:])
                                     for node in moving_control_nodes])

    # 如果指定了均匀节点数量，应用 select_uniform_nodes 来选择运动控制点
    if uniform_moving_node_count is not None:
        moving_control_nodes = select_uniform_nodes(moving_control_nodes, uniform_moving_node_count)
        moving_source_points = np.array([[node._id] + list(all_points[all_points[:, 0] == node._id][0, 1:]) 
                                        for node in moving_control_nodes])
        moving_target_points = np.array([[node._id] + list(rotated_all_points[rotated_all_points[:, 0] == node._id][0, 1:])
                                        for node in moving_control_nodes])
    #print(f"运动控制点ID: {[node._id for node in moving_control_nodes]}")

    # 合并固定和运动控制点
    source_control_points = np.vstack([fixed_points, moving_source_points])
    target_control_points = np.vstack([fixed_points, moving_target_points])

    return source_control_points, target_control_points

def main(params):
    # 直接从字典中提取各个参数
    motion_method = params['motion_method']
    motion_param = params['motion_param']
    rotation_angle = params['rotation_angle']
    rotate_axis = params['rotate_axis']
    center_set_ids = params['center_set_ids']
    control_fixed_method = params['control_fixed_method']
    control_fixed_param = params['control_fixed_param']
    control_moving_method = params['control_moving_method']
    control_moving_param = params['control_moving_param']
    transition_method = params['transition_method']
    transition_param = params['transition_param']
    boundary_set_ids = params['boundary_set_ids']

    # 获取固定控制点的数量（可选），默认为None（即不进行筛选）
    uniform_fixed_node_count = params.get('uniform_fixed_node_count', None)
    uniform_moving_node_count = params.get('uniform_moving_node_count', None)

    # 获取核函数配置
    kernel = params.get('kernel', None)  # 默认使用薄板样条
    kernel_params = params.get('kernel_params', {})  # 获取其他核函数参数

    # 计算几何中心
    print("计算几何中心...")
    center_nodes = get_nodes_from_set(center_set_ids)  # 获取计算几何中心所用的节点
    center = calculate_geometric_center(center_nodes)

    # 获取所有需要旋转的目标节点（运动部件）
    print("获取运动部件节点...")
    motion_nodes = get_all_nodes(motion_method, motion_param)  # 获取运动部件节点列表
    motion_points = np.array([[node._id] + list(node.position) for node in motion_nodes])

    # 旋转目标节点（运动部件）
    print(f"旋转节点 {rotation_angle}°...")
    rotated_points = rotate_nodes_from_set_optimized(motion_points, center, rotate_axis, rotation_angle)
    update_ansa_node_coordinates(rotated_points, motion_nodes)

    # 获取源控制点和目标控制点
    print("获取控制点...")
    source_control_points, target_control_points = get_control_points4(
        motion_points, rotated_points, control_fixed_method, control_fixed_param, control_moving_method, control_moving_param,
        uniform_fixed_node_count=uniform_fixed_node_count,  # 如果有，选择固定控制点
        uniform_moving_node_count=uniform_moving_node_count  # 如果有，选择运动控制点
    )

    # 获取需要排除的边界节点（边界部件）
    print("获取边界部件节点...")
    #获取过渡节点
    transition_nodes = get_all_nodes(transition_method, transition_param)  

    boundary_nodes = get_nodes_from_set(boundary_set_ids)  # 获取边界部件节点列表
    boundary_node_ids = [node._id for node in boundary_nodes]


    # 排除边界节点，得到需要过渡变换的节点（过渡部件）
    print("排除边界部件，得到过渡部件节点...")
    transition_filtered_nodes = [node for node in transition_nodes if node._id not in boundary_node_ids]  # 过渡部件节点
    transition_filtered_points = np.array([[node._id] + list(node.position) for node in transition_filtered_nodes])

    # 执行 RBF 变换，过渡部件
    print("执行 RBF 变换...")
    transformed_points = rbf_transform_3d_chunked(
        transition_filtered_points, source_control_points, target_control_points, 0, 20000,
        kernel=kernel, **kernel_params
    )
    update_ansa_node_coordinates(transformed_points, transition_filtered_nodes)


# 记录总的开始时间
start_time = time.time()
#wrist
params_wrist = {
    'motion_method': "pid",
    'motion_param': ["86201001"],
    'rotation_angle': 30,
    'rotate_axis': [0, 0, 1],
    'center_set_ids': ["33"],
    'control_fixed_method': "pid",
    'control_fixed_param': ["86200501"],
    'control_moving_method': "pid",
    'control_moving_param': ["86201001"],
    'boundary_set_ids': ["27", "28"],
    'transition_method': "pid",
    'transition_param': ["86200801"],    
    'uniform_fixed_node_count': None,  # 选择50个固定控制点   
    'uniform_moving_node_count': None,  # 不筛选运动控制点
}
main(params_wrist)

#Elbow
params_elbow = {
    'motion_method': "pid",
    'motion_param': ["86200501", "86200801", "86201001"],
    'rotation_angle': -20,
    'rotate_axis': [0, 1, 0],
    'center_set_ids': ["32"],
    'control_fixed_method': "pid",
    'control_fixed_param': ["86200001"],
    'control_moving_method': "pid",
    'control_moving_param': ["86200501"],
    'boundary_set_ids': ["24", "26"],
    'transition_method': "pid",
    'transition_param': ["86200301"],
    'uniform_fixed_node_count': None,  # 选择50个固定控制点   
    'uniform_moving_node_count': 50,  # 不筛选运动控制点
}
main(params_elbow)

# Shoulder
params_shoulder = {
    'motion_method': "pid",
    'motion_param': ["86200001", "86200301", "86200501", "86200801", "86201001"],
    'rotation_angle': -75,
    'rotate_axis': [1, 0, 0],
    'center_set_ids': ["31"],
    'control_fixed_method': "set",
    'control_fixed_param': ["50"],
    'control_moving_method': "set",
    'control_moving_param': ["52"],
    'boundary_set_ids': ["22", "20"],
    'transition_method': "pid",
    'transition_param': ["89200701"],
    # 'kernel': 'multiquadric', #可选 'linear'、'cubic'、'gaussian'、'multiquadric'
    # 'kernel_params': {'epsilon': 1}
}
# 调用main函数，只需要传递封装的参数字典
main(params_shoulder)

laplacian_smoothing(pids=["89200701"], iterations=10, alpha=0.01)
reflect(rules_to_run=["手部规则","肩胸规则"])
# 打印运行时间
end_time = time.time()
print(f"总运行时间: {end_time - start_time:.2f} s")