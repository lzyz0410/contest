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

# 命名规则：ansa实体对象用nodes;np.array含id和坐标用points

def process_nodes(nodes_method, nodes_param, filter_by_prefix=False,total_side_num=None, total_plane_num=None, control_source_param=None,
                  enforce_uniformity=False, enforce_params=None, laplacian_smooth=False, laplacian_params=None,
                  run_reflect=False, reflect_rules=None):
    """
    处理节点的主函数。

    参数：
        nodes_method (str): 获取 all_nodes 的方式（如 "pid", "set"）。
        nodes_param (list): 获取 all_nodes 的参数（如 PID 列表）。
        filter_by_prefix (bool): 是否根据节点 ID 的前两位进行过滤，默认为 False。
        total_side_num (int or None): 左右两侧控制点的总数量（两侧合计，不含对称面）。
        total_plane_num (int or None): 对称面控制点数量。
        control_source_param (list or None): 如果不使用自动选取，指定控制点的 set ID。
        enforce_uniformity (bool): 是否调用 enforce_coordinate_uniformity 函数。
        enforce_params (dict or None): enforce_coordinate_uniformity 的参数。
        laplacian_smooth (bool): 是否调用 laplacian_smoothing 函数。
        laplacian_params (dict or None): laplacian_smoothing 的参数。
        run_reflect (bool): 是否运行 reflect 函数。
        reflect_rules (list or None): 需要执行的 reflect 规则名称列表。
    """
    start_time = time.time()

    # 读取基准网格和目标网格的节点
    all_nodes = get_all_nodes(nodes_method, nodes_param,filter_by_prefix=filter_by_prefix)
    target_all_nodes = get_nodes_from_set([2])

    # 判断控制点获取方式
    if total_side_num is not None and total_plane_num is not None:  # 如果提供了左右两侧和对称面的数量
        control_source_nodes = select_symmetric_uniform_nodes(all_nodes, total_side_num, total_plane_num)
        #print(f"使用自动选取的控制点数量: 左右两侧合计 {total_side_num}，对称面 {total_plane_num}")
    elif control_source_param:  # 否则使用指定的 set
        control_source_nodes = get_nodes_from_set(control_source_param)
        #print(f"使用指定 set 的控制点，set ID: {control_source_param}")
    else:
        raise ValueError("需要提供 total_side_num 和 total_plane_num 或 control_source_param")

    # 转换控制点为 numpy 数组
    control_source_points = np.array([[node._id] + list(node.position) for node in control_source_nodes])
    #print(f"源控制点ID: {control_source_points[:, 0].astype(int).tolist()}")

    # 根据初始控制点搜索目标控制点
    control_target_points = compute_projection_points(control_source_nodes, target_all_nodes, alpha_selected=0.5)
    #print(f"目标控制点ID: {control_target_points[:, 0].astype(int).tolist()}")

    # 执行 RBF 插值变换并更新节点坐标
    all_points = np.array([[node._id] + list(node.position) for node in all_nodes])
    transformed_points = rbf_transform_3d_chunked(all_points, control_source_points, control_target_points, alpha=0)
    update_ansa_node_coordinates(transformed_points, all_nodes)

    # 根据任务需求调用 enforce_coordinate_uniformity
    if enforce_uniformity and enforce_params:
        set_nodes = get_nodes_from_set(enforce_params.get("set_ids", []))  # 获取对称面节点
        left_pids = enforce_params.get("left_pids", []) or []  # 如果 left_pids 为 None，则返回空列表
        left_nodes = get_all_nodes("pid", left_pids)
        enforce_coordinate_uniformity(
            set_nodes,
            axis=enforce_params.get("axis", "y"),  # 默认对齐 Y 轴
            left_nodes=left_nodes,
            smoothing_factor=enforce_params.get("smoothing_factor", 0.5)  # 默认平滑因子 0.5
        )
        print("已调用 enforce_coordinate_uniformity 函数")

    # 根据任务需求调用 laplacian_smoothing
    if laplacian_smooth and laplacian_params:
        smooth_pids = laplacian_params.get("node_pids", [])  # 获取需要平滑的节点的 PID 列表
        laplacian_smoothing(
            pids=smooth_pids,  # 传递 PID 列表
            iterations=laplacian_params.get("iterations", 1),  # 默认迭代次数 1
            alpha=laplacian_params.get("alpha", 0)  # 默认 alpha 值 0
        )
        print("已调用 laplacian_smoothing 函数")

    # 根据任务需求调用 reflect
    if run_reflect:
        print("开始运行 reflect...")
        if reflect_rules:
            # 如果指定了规则，则执行指定的规则
            reflect(rules_to_run=reflect_rules)
        else:
            # 否则执行所有规则
            reflect(run_all_rules=True)
        print("reflect 运行完成")

    print(f"耗时: {time.time() - start_time:.2f} 秒\n")


def main():
    # 批量输入的参数配置
    batch_configs = [
        {
            "nodes_method": "pid",
            "nodes_param": ["88000222", "88000230"],
            "total_side_num": 40,  # 左右两侧控制点的总数量
            "total_plane_num": 20,  # 对称面控制点数量
            "enforce_uniformity": True,  # 调用 enforce_coordinate_uniformity
            "enforce_params": {  # enforce_coordinate_uniformity 的参数
                "set_ids": [3, 4, 5, 6],  # 对称面节点的 set ID
                "left_pids": ["88000230"],  # 左侧节点的 PID 列表
                "axis": "y",  # 对齐 Y 轴
                "smoothing_factor": 0.5  # 平滑因子
            },
            "laplacian_smooth": True,  # 调用 laplacian_smoothing
            "laplacian_params": {  # laplacian_smoothing 的参数
                "node_pids": ["88000230"],  # 需要平滑的节点的 PID 列表
                "iterations": 1,  # 迭代次数
                "alpha": 0  # alpha 值
            },
            "run_reflect": True,  # 运行 reflect
            "reflect_rules": ["头部规则"]  # 指定需要执行的规则
        },
        {
            "nodes_method": "pid",
            "nodes_param": ["88000229", "88000231","88000221","88000223"],
            "total_side_num": 40,  # 左右两侧控制点的总数量
            "total_plane_num": 20,  # 对称面控制点数量
            "enforce_uniformity": True,  # 调用 enforce_coordinate_uniformity
            "enforce_params": {  # enforce_coordinate_uniformity 的参数
                "set_ids": [3, 4, 5, 6],  # 对称面节点的 set ID
                "left_pids": ["88000229", "88000231"],  # 左侧节点的 PID 列表
                "axis": "y",  # 对齐 Y 轴
                "smoothing_factor": 0.5  # 平滑因子
            },
            "laplacian_smooth": True,  # 调用 laplacian_smoothing
            "laplacian_params": {  # laplacian_smoothing 的参数
                "node_pids": ["88000229", "88000231"],  # 需要平滑的节点的 PID 列表
                "iterations": 1,  # 迭代次数
                "alpha": 0  # alpha 值
            },
            "run_reflect": True,  # 运行 reflect
            "reflect_rules": ["头部规则"]  # 指定需要执行的规则
        },
        {
            "nodes_method": "pid",
            "nodes_param": ["87200101", "87700101"],
            "filter_by_prefix": True,  # 根据节点 ID 的前两位进行过滤
            "total_side_num": 30,  # 左右两侧控制点的总数量
            "total_plane_num": 15,  # 对称面控制点数量
            "enforce_uniformity": True,  # 调用 enforce_coordinate_uniformity
            "enforce_params": {  # enforce_coordinate_uniformity 的参数
                "set_ids": [17],  # 对称面节点的 set ID
                "left_pids": None,  # 左侧节点的 PID 列表
                "axis": "o",  
                "smoothing_factor": 0.5  # 平滑因子
            },
            "laplacian_smooth": True,  # 调用 laplacian_smoothing
            "laplacian_params": {  # laplacian_smoothing 的参数
                "node_pids": ["87200101"],  # 需要平滑的节点的 PID 列表
                "iterations": 2,  # 迭代次数
                "alpha": 0  # alpha 值
            },
            "run_reflect": True,  # 运行 reflect
            "reflect_rules": ["颈部规则"]  # 指定需要执行的规则
        },
        {
            "nodes_method": "pid",
            "nodes_param": ["89200801", "89700801"],
            "filter_by_prefix": True,  # 根据节点 ID 的前两位进行过滤
            "total_side_num": 200,  # 左右两侧控制点的总数量
            "total_plane_num": 100,  # 对称面控制点数量
            "enforce_uniformity": True,  # 调用 enforce_coordinate_uniformity
            "enforce_params": {  # enforce_coordinate_uniformity 的参数
                "set_ids": [3, 4, 5, 6],  # 对称面节点的 set ID
                "left_pids": ["89200801"],  # 左侧节点的 PID 列表
                "axis": "y",  # 对齐 Y 轴
                "smoothing_factor": 0.5  # 平滑因子
            },
            "laplacian_smooth": True,  # 调用 laplacian_smoothing
            "laplacian_params": {  # laplacian_smoothing 的参数
                "node_pids": ["89200801"],  # 需要平滑的节点的 PID 列表
                "iterations": 5,  # 迭代次数
                "alpha": 0  # alpha 值
            },
            "run_reflect": True,  # 运行 reflect
            "reflect_rules": ["肩胸规则"]  # 指定需要执行的规则
        },
        {
            "nodes_method": "pid",
            "nodes_param": ["89200701", "89700701"],
            "total_side_num": 300,  # 左右两侧控制点的总数量
            "total_plane_num": 0,  # 对称面控制点数量
            "enforce_uniformity": True,  # 调用 enforce_coordinate_uniformity
            "enforce_params": {  # enforce_coordinate_uniformity 的参数
                "set_ids": [20],  # 对称面节点的 set ID
                "left_pids": None,  # 左侧节点的 PID 列表
                "axis": "o",  
                "smoothing_factor": 0.5  # 平滑因子
            },
            "laplacian_smooth": True,  # 调用 laplacian_smoothing
            "laplacian_params": {  # laplacian_smoothing 的参数
                "node_pids": ["89200701"],  # 需要平滑的节点的 PID 列表
                "iterations":10,  # 迭代次数
                "alpha": 0  # alpha 值
            },
            "run_reflect": True,  # 运行 reflect
            "reflect_rules": ["肩胸规则"]  # 指定需要执行的规则
        },         
        {
            "nodes_method": "pid",
            "nodes_param": ["83200101", "83700101"],
            "total_side_num": 20,  # 左右两侧控制点的总数量
            "total_plane_num": 10,  # 对称面控制点数量
            "enforce_uniformity": True,  # 调用 enforce_coordinate_uniformity
            "enforce_params": {  # enforce_coordinate_uniformity 的参数
                "set_ids": [15],  # 对称面节点的 set ID
                "left_pids": ["83200101"],  # 左侧节点的 PID 列表
                "axis": "z",  # 对齐 z 轴
                "smoothing_factor": 0.5  # 平滑因子
            },
            "laplacian_smooth": True,  # 调用 laplacian_smoothing
            "laplacian_params": {  # laplacian_smoothing 的参数
                "node_pids": ["83200101"],  # 需要平滑的节点的 PID 列表
                "iterations": 5,  # 迭代次数
                "alpha": 0  # alpha 值
            },
            "run_reflect": True,  # 运行 reflect
            "reflect_rules": ["臀部规则"]  # 指定需要执行的规则
        },
        { #大腿
            "nodes_method": "pid",  #
            "nodes_param": ["82200001", "81200001"],
            "filter_by_prefix": True,  # 根据节点 ID 的前两位进行过滤
            "total_side_num": 40,  # 左右两侧控制点的总数量
            "total_plane_num": 0,  # 对称面控制点数量
            "enforce_uniformity": True,  # 调用 enforce_coordinate_uniformity
            "enforce_params": {  # enforce_coordinate_uniformity 的参数
                "set_ids": [14],  # 对称面节点的 set ID
                "left_pids": ["82200001"],  # 左侧节点的 PID 列表
                "axis": "z",  # 对齐 z 轴
                "smoothing_factor": 0.5  # 平滑因子
            },
            "laplacian_smooth": True,  # 调用 laplacian_smoothing
            "laplacian_params": {  # laplacian_smoothing 的参数
                "node_pids": ["82200001"],  # 需要平滑的节点的 PID 列表
                "iterations": 5,  # 迭代次数
                "alpha": 0  # alpha 值
            },
            "run_reflect": True,  # 运行 reflect
            "reflect_rules": ["腿部规则"]  # 指定需要执行的规则
        },
        { #膝盖
            "nodes_method": "pid",
            "nodes_param": ["82200401","81200401"],
            "total_side_num": 200,  # 左右两侧控制点的总数量
            "total_plane_num": 0,  # 对称面控制点数量
            "enforce_uniformity": True,  # 调用 enforce_coordinate_uniformity
            "enforce_params": {  # enforce_coordinate_uniformity 的参数
                "set_ids": [12],  # 对称面节点的 set ID
                "left_pids": ["82200401"],  # 左侧节点的 PID 列表
                "axis": "z",  # 对齐 z 轴
                "smoothing_factor": 0.5  # 平滑因子
            },
            "laplacian_smooth": True,  # 调用 laplacian_smoothing
            "laplacian_params": {  # laplacian_smoothing 的参数
                "node_pids": ["82200401"],  # 需要平滑的节点的 PID 列表
                "iterations": 5,  # 迭代次数
                "alpha": 0  # alpha 值
            },
            "run_reflect": True,  # 运行 reflect
            "reflect_rules": ["腿部规则"]  # 指定需要执行的规则
        },
        { #小腿
            "nodes_method": "pid",
            "nodes_param": ["82200601","81200601"],
            "total_side_num": 100,  # 左右两侧控制点的总数量
            "total_plane_num": 0,  # 对称面控制点数量
            "enforce_uniformity": True,  # 调用 enforce_coordinate_uniformity
            "enforce_params": {  # enforce_coordinate_uniformity 的参数
                "set_ids": [10],  # 对称面节点的 set ID
                "left_pids": ["82200601"],  # 左侧节点的 PID 列表
                "axis": "z",  # 对齐 z 轴
                "smoothing_factor": 0.5  # 平滑因子
            },
            "laplacian_smooth": True,  # 调用 laplacian_smoothing
            "laplacian_params": {  # laplacian_smoothing 的参数
                "node_pids": ["82200601"],  # 需要平滑的节点的 PID 列表
                "iterations": 5,  # 迭代次数
                "alpha": 0  # alpha 值
            },
            "run_reflect": True,  # 运行 reflect
            "reflect_rules": ["腿部规则"]  # 指定需要执行的规则
        },
        { #脚踝
            "nodes_method": "pid",
            "nodes_param": ["82201101","81201101"],
            "total_side_num": 50,  # 左右两侧控制点的总数量
            "total_plane_num": 0,  # 对称面控制点数量
            "enforce_uniformity": True,  # 调用 enforce_coordinate_uniformity
            "enforce_params": {  # enforce_coordinate_uniformity 的参数
                "set_ids": [8],  # 对称面节点的 set ID
                "left_pids": ["82201101"],  # 左侧节点的 PID 列表
                "axis": "z",  # 对齐 z 轴
                "smoothing_factor": 0.5  # 平滑因子
            },
            "laplacian_smooth": True,  # 调用 laplacian_smoothing
            "laplacian_params": {  # laplacian_smoothing 的参数
                "node_pids": ["82201101"],  # 需要平滑的节点的 PID 列表
                "iterations": 5,  # 迭代次数
                "alpha": 0  # alpha 值
            },
            "run_reflect": True,  # 运行 reflect
            "reflect_rules": ["腿部规则"]  # 指定需要执行的规则
        },
        { #脚
            "nodes_method": "pid",
            "nodes_param": ["82201301", "81201301"],
            "total_side_num": 80,  # 左右两侧控制点的总数量
            "total_plane_num": 0,  # 对称面控制点数量
            "laplacian_smooth": True,  # 调用 laplacian_smoothing
            "laplacian_params": {  # laplacian_smoothing 的参数
                "node_pids": ["82201301"],  # 需要平滑的节点的 PID 列表
                "iterations": 1,  # 迭代次数
                "alpha": 0  # alpha 值
            },
            "run_reflect": True,  # 运行 reflect
            "reflect_rules": ["腿部规则"]  # 指定需要执行的规则
        },
        { #大臂
            "nodes_method": "pid",
            "nodes_param": ["86200001","85200001"],
            "filter_by_prefix": True,  # 根据节点 ID 的前两位进行过滤
            "total_side_num": 200,  # 左右两侧控制点的总数量
            "total_plane_num": 0,  # 对称面控制点数量
            "enforce_uniformity": True,  # 调用 enforce_coordinate_uniformity
            "enforce_params": {  # enforce_coordinate_uniformity 的参数
                "set_ids": [22],  # 对称面节点的 set ID
                "left_pids": None,  # 左侧节点的 PID 列表
                "axis": "o",  
                "smoothing_factor": 0.5  # 平滑因子
            },
            "laplacian_smooth": True,  # 调用 laplacian_smoothing
            "laplacian_params": {  # laplacian_smoothing 的参数
                "node_pids": ["86200001"],  # 需要平滑的节点的 PID 列表
                "iterations": 1,  # 迭代次数
                "alpha": 0  # alpha 值
            },
            "run_reflect": True,  # 运行 reflect
            "reflect_rules": ["手部规则"]  # 指定需要执行的规则
        },
        {#肘关节
            "nodes_method": "pid",
            "nodes_param": ["86200301", "85200301"],
            "total_side_num": 50,  # 左右两侧控制点的总数量
            "total_plane_num": 0,  # 对称面控制点数量
            "enforce_uniformity": True,  # 调用 enforce_coordinate_uniformity
            "enforce_params": {  # enforce_coordinate_uniformity 的参数
                "set_ids": [24],  # 对称面节点的 set ID
                "left_pids": None,  # 左侧节点的 PID 列表
                "axis": "o",  
                "smoothing_factor": 0.5  # 平滑因子
            },
            "laplacian_smooth": True,  # 调用 laplacian_smoothing
            "laplacian_params": {  # laplacian_smoothing 的参数
                "node_pids": ["86200301"],  # 需要平滑的节点的 PID 列表
                "iterations": 1,  # 迭代次数
                "alpha": 0  # alpha 值
            },
            "run_reflect": True,  # 运行 reflect
            "reflect_rules": ["手部规则"]  # 指定需要执行的规则
        },
        {#小臂
            "nodes_method": "pid",
            "nodes_param": ["86200501", "85200501"],
            "total_side_num": 50,  # 左右两侧控制点的总数量
            "total_plane_num": 0,  # 对称面控制点数量
            "laplacian_smooth": True,  # 调用 laplacian_smoothing
            "laplacian_params": {  # laplacian_smoothing 的参数
                "node_pids": ["86200501"],  # 需要平滑的节点的 PID 列表
                "iterations": 1,  # 迭代次数
                "alpha": 0  # alpha 值
            },
            "run_reflect": True,  # 运行 reflect
            "reflect_rules": ["手部规则"]  # 指定需要执行的规则
        },
        # {#踝手
        #     "nodes_method": "pid",
        #     "nodes_param": ["86200801","86201001","85201001","85200801"],
        #     "total_side_num": 100,  # 左右两侧控制点的总数量
        #     "total_plane_num": 0,  # 对称面控制点数量
        #     # "enforce_uniformity": True,  # 调用 enforce_coordinate_uniformity
        #     # "enforce_params": {  # enforce_coordinate_uniformity 的参数
        #     #     "set_ids": [27],  # 对称面节点的 set ID
        #     #     "left_pids": None,  # 左侧节点的 PID 列表
        #     #     "axis": "o",  
        #     #     "smoothing_factor": 0.5  # 平滑因子
        #     # },
        #     # "laplacian_smooth": True,  # 调用 laplacian_smoothing
        #     # "laplacian_params": {  # laplacian_smoothing 的参数
        #     #     "node_pids": ["86200801", "86201001"],  # 需要平滑的节点的 PID 列表
        #     #     "iterations": 1,  # 迭代次数
        #     #     "alpha": 0  # alpha 值
        #     # },
        #     # "run_reflect": True,  # 运行 reflect
        #     # "reflect_rules": ["手臂规则"]  # 指定需要执行的规则
        # },
    ]

    for config in batch_configs:
        process_nodes(
            config["nodes_method"],
            config["nodes_param"],
            config.get("filter_by_prefix", False),  # 默认不启用前缀过滤
            config.get("total_side_num"),  # 使用 get 方法以支持 None
            config.get("total_plane_num"),  # 使用 get 方法以支持 None
            config.get("control_source_param"),  # 使用 get 方法以支持 None
            config.get("enforce_uniformity", False),  # 默认不调用 enforce_coordinate_uniformity
            config.get("enforce_params"),  # enforce_coordinate_uniformity 的参数
            config.get("laplacian_smooth", False),  # 默认不调用 laplacian_smoothing
            config.get("laplacian_params"),  # laplacian_smoothing 的参数
            config.get("run_reflect", False),  # 默认不运行 reflect
            config.get("reflect_rules")  # 指定需要执行的规则
        )


if __name__ == '__main__':
    start_time = time.time()
    main()
    reflect(rules_to_run=True)
    end_time = time.time()
    print(f"所有处理完成，耗时: {end_time - start_time:.2f} s")