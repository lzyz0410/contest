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

#命名规则ansa实体对象用nodes;np.array含id和坐标用points


def main():
    start_time = time.time()

    #读取指定部分的所有点
    all_nodes = get_all_nodes("set", ["41", "42"])
    all_points = np.array([[node._id] + list(node.position) for node in all_nodes])

    #读取目标网格的所有点
    target_all_nodes = get_nodes_from_set([2])

    uniform_node_count = 10
    #从基准网格中均匀选取控制点
    control_source_nodes = select_symmetric_uniform_nodes(all_nodes, uniform_node_count)
    control_source_points = np.array([[node._id] + list(node.position) for node in control_source_nodes])

    print(f"源控制点ID: {control_source_points[:, 0].astype(int).tolist()}")

    #根据初始控制点搜索目标控制点
    control_target_points = compute_projection_points(control_source_nodes, target_all_nodes, alpha_selected=0.5)
    print(f"目标控制点ID: {control_target_points[:, 0].astype(int).tolist()}")

    #执行 RBF 插值变换...")
    transformed_points = rbf_transform_3d_chunked(all_points, control_source_points, control_target_points, alpha=0)

    # 更新 ANSA 中的节点坐标
    update_ansa_node_coordinates(transformed_points, all_nodes)

    end_time = time.time()
    print(f"所有处理完成，耗时: {end_time - start_time:.2f} 秒")


if __name__ == '__main__':
    main()