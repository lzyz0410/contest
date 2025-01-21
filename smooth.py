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
from scipy.optimize import curve_fit

# 最小二乘拟合
def least_squares_fitting(points):
    # 定义拟合函数
    def model(X, a, b, c, d, e, f):
        x, y = X
        return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f
    
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # 使用curve_fit拟合数据
    # 注意：curve_fit期望传递一个二维数组（x, y）给模型函数
    params, _ = curve_fit(model, (x, y), z)
    
    # 用拟合模型对数据进行平滑
    smoothed_z = model((x, y), *params)
    smoothed_points = np.column_stack((x, y, smoothed_z))
    
    return smoothed_points


# 获取节点数据
nodes_method = "pid"
nodes_param = ["89200701"]
all_nodes1 = get_all_nodes(nodes_method, nodes_param)
all_points1 = np.array([[node._id] + list(node.position) for node in all_nodes1])

nodes_method = "pid"
nodes_param = ["86200001"]
all_nodes2 = get_all_nodes(nodes_method, nodes_param)
all_points2 = np.array([[node._id] + list(node.position) for node in all_nodes2])

# 合并两个曲面的数据（all_points1 和 all_points2）
all_points_combined = np.vstack((all_points1[:, 1:], all_points2[:, 1:]))

# 对合并后的数据进行平滑
smoothed_combined_points = least_squares_fitting(all_points_combined)

# 输出平滑后的 np.array，含节点 id 和坐标
smoothed_all_points = np.column_stack((all_points_combined[:, 0], smoothed_combined_points))  # 组合id和坐标

# 将平滑后的结果分为两个部分
smoothed_points1 = smoothed_all_points[:len(all_points1), :]  # 第一个曲面
smoothed_points2 = smoothed_all_points[len(all_points1):, :]  # 第二个曲面

# 更新平滑后的节点坐标
update_ansa_node_coordinates(smoothed_points1, all_nodes1)
update_ansa_node_coordinates(smoothed_points2, all_nodes2)
