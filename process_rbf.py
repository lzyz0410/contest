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
from utils_reflect import *


#命名规则ansa实体对象用nodes;np.array含id和坐标用points

def get_control_points1(control_points_csv_path):
    # 读取基准控制点的节点 ID 和目标控制点的节点 ID
    source_control_nodeids = read_csv_single_column(control_points_csv_path, "B2:B")# 基准控制点的节点 ID
    target_control_nodeids = read_csv_single_column(control_points_csv_path, "C2:C")# 目标控制点的节点 ID

    #print(f"基准控制点的节点 ID 列表: {base_nodeids}, len={len(base_nodeids)}")
    #print(f"目标控制点的节点 ID 列表: {target_nodeids}, len={len(target_nodeids)}") 

    # 使用 get_nodes_from_ids 获取基准和目标控制点的坐标
    source_control_nodes = get_nodes_from_ids(source_control_nodeids)  # 基准控制点的节点对象
    target_control_nodes = get_nodes_from_ids(target_control_nodeids)  # 目标控制点的节点对象

    # 提取节点坐标
    source_control_points = np.array([[node._id] + list(node.position) for node in source_control_nodes])
    target_control_points = np.array([[node._id] + list(node.position) for node in target_control_nodes])
    return source_control_points, target_control_points

def main():
    start_time = time.time()    # 定义多个 control_points_csv_path

    #根据配置获取目标节点
    nodes_method = "csv"  # 获取节点的方式：从 CSV 获取 PID 列表
    nodes_param = "E:\\LZYZ\\Scoliosis\\RBF\\Contest\\final\\shell_property.csv"
    range = "A2:B35"  # 读取 PID 的范围
    all_nodes = get_all_nodes(nodes_method, nodes_param,range)
    
    all_points = np.array([[node._id] + list(node.position) for node in all_nodes])

    source_control_points, target_control_points = get_control_points1("E:\\LZYZ\\Scoliosis\\RBF\\Contest\\final\\point1all.csv")

    # 执行 RBF 变换
    transformed_all_points = rbf_transform_3d_chunked(all_points, source_control_points, target_control_points,0,20000)

    # 更新 ANSA 中的节点坐标
    update_ansa_node_coordinates(transformed_all_points, all_nodes)

    
    # nodes_method = "pid"  # 获取节点的方式
    # nodes_param = ["89200701","86200001", "86200301", "86200501", "86200801", "86201001", 
    #                "89700701","85200001", "85200301", "85200501", "85200801", "85201001"]  # 节点 ID 列表
    # all_nodes = get_all_nodes(nodes_method, nodes_param)
    
    # all_points = np.array([[node._id] + list(node.position) for node in all_nodes])

    # source_control_points, target_control_points = get_control_points1("E:\\LZYZ\\Scoliosis\\RBF\\Contest\\final\\point1_hand.csv")

    # # 执行 RBF 变换
    # transformed_all_points = rbf_transform_3d_chunked(all_points, source_control_points, target_control_points,0,20000)

    # # 更新 ANSA 中的节点坐标
    # update_ansa_node_coordinates(transformed_all_points, all_nodes)


    end_time = time.time()
    print(f"所有处理完成，耗时: {end_time - start_time:.2f} s")


    # #test
    # input_file_base = "E:\\LZYZ\\Scoliosis\\RBF\\Contest\\final\\THUMS_AM50_V402_Pedestrian_20150527_no_fracture2.k"
    # updated_node_data = [
    #     Node(81000001, [5.0, 0.1, 0.2]),
    #     Node(81000002, [1.0, 1.1, 1.2]),
    #     Node(81000003, [2.0, 2.1, 2.2])
    # ]
    # # 加载原始文件内容并写入修改后的数据
    # with open(input_file_base, "r") as f:
    #     file_lines = f.readlines()

    # write_modified_coordinates(
    #     output_file="E:\\LZYZ\\Scoliosis\\RBF\\Contest\\final\\output.k",
    #     file_lines=file_lines,
    #     updated_node_data=updated_node_data
    # )
    # print("修改后的坐标已写入文件：output.k")


if __name__ == "__main__":
    main()
    reflect(run_all_rules=True)