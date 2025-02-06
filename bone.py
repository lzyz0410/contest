import os
import sys
third_packages = r"G:\\pyhton3119\\Lib\\site-packages"
sys.path.append(third_packages)

import ansa
from ansa import *
import numpy as np
import time
import pandas as pd
from utils_data import *
from utils_node import *
from utils_rbf_transform import *
from utils_env import *
from utils_bond_locate import *
from utils_reflect import *
# 全局字典
node_id_to_transformed_coords = {}
from pathlib import Path
input_file = find_file_in_parents("output.k")
output_file = input_file.with_name(input_file.stem + "_transformed.k")
shell_property_file = find_file_in_parents("shell_property.csv")
file_lines = None

def process_bone_locating_points(task_settings):
    """
    主任务函数：处理并变换骨架定位点
    1. 从 `initialize_shell_data` 获取所有的节点和原始坐标。
    2. 使用 `project_and_transform_to_surface` 计算每个方向的投影外部点和变换后的定位点。
    3. 输出变换后的定位点供后续使用。
    """
    # Step 1: 初始化所有表面节点数据
    print("初始化骨架定位点数据...")
    all_shell_points, node_id_to_transformed_coords, file_lines = initialize_shell_data()
    # 获取当前任务的设置
    use_manual_source_nodes = task_settings.get("use_manual_source_nodes", False) #false是用xyz投影点
    transform_sets = task_settings["transform_sets"]  # 将集合转换为列表
    source_shell_nodes = task_settings["source_shell_nodes"]  # 假设已经是字符串列表
    total_side_num = task_settings.get("total_side_num", 0)  # 如果键不存在，返回 0
    total_plane_num = task_settings.get("total_plane_num", 0)  # 如果键不存在，返回 0
    modify_nodes = task_settings.get("modify_nodes", [])  # 用于修改特定节点的变换后的内部点
    node_direction_map = task_settings.get("node_direction_map", {})

    # 获取 transform_points
    transform_nodes = get_all_nodes("set", transform_sets)
    transform_points = np.array([[node._id] + list(node.position) for node in transform_nodes])

    # 获取控制点
    source_shell_nodes = get_all_nodes("pid", source_shell_nodes)
    if total_side_num == 0:
        print("SS`total_side_num` 和 `total_plane_num` 均为 0，直接使用 `source_shell_nodes` 作为 `source_select_nodes`")
        source_select_points = np.empty((0, 4))  # **确保是二维空数组**
        target_select_points = np.empty((0, 4))
    else:
        source_select_nodes = select_symmetric_uniform_nodes(source_shell_nodes, total_side_num, total_plane_num)
        source_select_points = np.array([[node._id] + list(node.position) for node in source_select_nodes])
        # 获取目标坐标
        target_select_points = get_target_coordinates(source_select_points)

    if use_manual_source_nodes:
        # print("`use_manual_source_nodes=True`，不进行投影，仅使用 `source_shell_nodes` 进行 RBF 变换")
        # print("source_select_points")
        # print([int(point[0]) for point in source_select_points])
        # print("target_select_points")
        # print([int(point[0]) for point in target_select_points])
        # 进行 RBF 变换
        after_transformed_points = rbf_transform_3d_chunked(
            transform_points,
            source_select_points,
            target_select_points,
            0
        )
    else:
        print("`use_manual_source_nodes=False`，使用 XYZ 投影点计算 RBF 控制点")
        # 获取投影后的内部和外部定位点
        inner_locating_points, outer_locating_points, transformed_outer_locating_points, transformed_inner_locating_points = project_and_transform_to_surface(all_shell_points, node_direction_map)

        # **如果 `modify_nodes` 不为空，则进行修改**
        if modify_nodes:
            print("执行 modify_transformed_inner_points")
            transformed_inner_locating_points = modify_transformed_inner_points(transformed_inner_locating_points, modify_nodes)

        # print("inner_locating_points")
        # print([int(point[0]) for point in inner_locating_points])
        # print("outer_locating_points")
        # print([int(point[0]) for point in outer_locating_points])

        # 合并内部、外部和变换后的定位点
        # 使用 numpy 的 unique 方法去重

        # 合并所有 source points
        source_all_points = np.vstack([
            source_select_points,
            inner_locating_points,
            outer_locating_points
        ])
        # print("source_all_points")
        # # 直接打印所有的 node_id，格式为 int[node_id]
        # print([int(point[0]) for point in source_all_points])
        # 合并所有 target points
        target_all_points = np.vstack([
            target_select_points,
            transformed_inner_locating_points,
            transformed_outer_locating_points
        ])
        # print("target_all_points")
        # print([int(point[0]) for point in target_all_points])
        # 使用 numpy.unique 来确保 node_id 不重复
        _, unique_source_idx = np.unique(source_all_points[:, 0], return_index=True)
        _, unique_target_idx = np.unique(target_all_points[:, 0], return_index=True)

        source_control_points = source_all_points[unique_source_idx]
        target_control_points = target_all_points[unique_target_idx]
        
        # 进行 RBF 变换
        after_transformed_points = rbf_transform_3d_chunked(
            transform_points,
            source_control_points,
            target_control_points,
            0
        )
        update_ansa_node_coordinates(after_transformed_points, transform_nodes)

    return after_transformed_points,file_lines



def get_target_coordinates_from_dict(points, transformed_points_dict):
    """
    从字典中获取变换后的坐标
    
    :param points: 包含节点 ID 和坐标的 numpy 数组，形状为 (n, 4)，每行格式为 [id, x, y, z]
    :param transformed_points_dict: 字典，键是节点 ID，值是变换后的坐标
    :return: 变换后的坐标
    """
    target_points = []
    
    for point in points:
        node_id = int(point[0])  # 获取节点 ID
        if node_id in transformed_points_dict:
            # 从字典中获取变换后的坐标并添加到目标点列表
            target_points.append([node_id] + list(transformed_points_dict[node_id]))
        else:
            # 如果未找到变换后的坐标，输出警告并跳过该节点
            print(f"警告：未找到节点 ID {node_id} 的变换坐标，跳过该节点。")
            continue  # 跳过该节点

    return np.array(target_points, dtype=float)


def process_inner(task2_settings, after_bone_transformed_points):
    # 获取当前任务的设
    inner_method = task2_settings.get("inner_method")
    inner_param = task2_settings.get("inner_param")
    source_shell_method = task2_settings.get("source_shell_method")
    source_shell_param = task2_settings.get("source_shell_param")
    source_bone_method = task2_settings.get("source_bone_method")
    source_bone_param = task2_settings.get("source_bone_param")
    shell_total_num = task2_settings.get("shell_total_num")
    bone_total_num = task2_settings.get("bone_total_num")

    # 获取反射规则（如果存在）
    rules_to_run = task2_settings.get("rules_to_run", None)

    # 获取内部节点
    print("正在获取内部节点...")
    inner_transformed_nodes = get_all_nodes(inner_method, inner_param)
    inner_transformed_points = np.array([[node._id] + list(node.position) for node in inner_transformed_nodes])

    print("正在获取源外壳节点...")
    source_shell_nodes = get_all_nodes(source_shell_method, source_shell_param)
    source_shell_nodes_selected_nodes = select_uniform_nodes(source_shell_nodes, shell_total_num)
    source_shell_nodes_selected_points = np.array([[node._id] + list(node.position) for node in source_shell_nodes_selected_nodes])
    print("source_shell_nodes_selected_points")
    print([int(point[0]) for point in source_shell_nodes_selected_points])
    
    # 使用 tqdm 显示进度条
    print("正在获取目标外壳节点坐标...")
    traget_shell_nodes_selected_points = get_target_coordinates(source_shell_nodes_selected_points)

    # 获取源骨架节点（set）
    print("正在获取源骨架节点...")
    source_bone_nodes = get_all_nodes(source_bone_method, source_bone_param)
    source_bone_nodes_selected_nodes = select_uniform_nodes(source_bone_nodes, bone_total_num)
    source_bone_nodes_selected_points = np.array([[node._id] + list(node.position) for node in source_bone_nodes_selected_nodes])
    print("source_bone_nodes_selected_points")
    print([int(point[0]) for point in source_bone_nodes_selected_points])

    # 获取变换后的坐标字典
    print("正在获取变换后的坐标字典...")
    transformed_points_dict = {int(point[0]): point[1:] for point in after_bone_transformed_points}

    # 从字典中获取变换后的坐标
    print("正在从字典中获取变换后的坐标...")
    target_bone_nodes_selected_points = get_target_coordinates_from_dict(
        source_bone_nodes_selected_points, transformed_points_dict)
    
    # 合并源点和目标点
    print("正在合并源点和目标点...")
    source_points = np.vstack([source_shell_nodes_selected_points, source_bone_nodes_selected_points])
    target_points = np.vstack([traget_shell_nodes_selected_points, target_bone_nodes_selected_points])

    # 执行 RBF 变换
    print("正在执行 RBF 变换...")
    after_inner_transformed_points = rbf_transform_3d_chunked(inner_transformed_points,source_points,target_points,0,chunk_size=1000)

    update_ansa_node_coordinates(after_inner_transformed_points, inner_transformed_nodes)
    if rules_to_run:
        print("正在执行反射规则...")
        reflected_points = reflect(rules_to_run)
         # 检查 reflected_points 的形状
        if reflected_points.size == 0:
            print("警告：反射规则未修改任何节点。")
        elif reflected_points.shape[1] != 4:
            print(f"错误：reflected_points 的形状为 {reflected_points.shape}，不符合 (n, 4) 的要求。")
            raise ValueError("reflected_points 的形状必须为 (n, 4)。")
        after_inner_transformed_points = np.vstack([after_inner_transformed_points, reflected_points])

    return after_inner_transformed_points, after_bone_transformed_points


def main(task_settings_list,task2_settings_list):

    start_time = time.time()
    all_transformed_inner_points = []
    # 检查 task_settings_list 是否只有一个任务
    if len(task_settings_list) != 1:
        raise ValueError("task_settings_list 必须包含且仅包含一个任务。")

    # 处理唯一的 task_settings
    print("处理骨架定位点变换...")
    task_settings = task_settings_list[0]
    after_bone_transformed_points, file_lines = process_bone_locating_points(task_settings)
    print("骨架定位点变换处理完成。")

    # 依次处理所有的 task2_settings
    for i, task2_settings in enumerate(task2_settings_list):
        print(f"\n正在处理第 {i + 1} 个 task2 任务...")
        
        # 处理内部节点变换
        print("处理内部节点变换...")
        after_inner_transformed_points, _ = process_inner(task2_settings, after_bone_transformed_points)

        # 将结果添加到全局列表中
        all_transformed_inner_points.append(after_inner_transformed_points)

        print(f"第 {i + 1} 个 task2 任务处理完成。")
    
    # 合并所有变换后的点
    # 将 after_bone_transformed_points 转换为 numpy 数组
    after_bone_transformed_points_array = np.array([[point[0]] + list(point[1:]) for point in after_bone_transformed_points])
    
    # 合并所有 after_inner_transformed_points
    all_inner_points = np.concatenate(all_transformed_inner_points, axis=0)
    
    # 合并 after_bone_transformed_points 和所有 after_inner_transformed_points
    updated_node_data = np.concatenate([all_inner_points, after_bone_transformed_points_array], axis=0)
    write_modified_coordinates(
        output_file=output_file,
        file_lines=file_lines,
        updated_node_data=updated_node_data,
    )
    print(f"所有任务完成，修改后的坐标已写入文件: {output_file}")

    end_time = time.time()
    print(f"文件名: {os.path.basename(__file__)}, 总运行时间: {end_time - start_time:.2f} s")

if __name__ == '__main__':
    # 示例任务配置
    task_settings_list = [
        {
            "node_direction_map": {
                'Xnegative': [(83010467, 'coccyx'),(89066294, 'Base of sacrum'),(89066279, 'Promontory'),(83012513,'ASIS-left'),(83512514,'ASIS-right'),
                (89004125,'Manubrium'), (89003192, 'xiphisternum'), (89059500, 'T1Am'),
                (89034241, 'Rib10LA'),(89034350, 'Rib8LA'), (89019613, 'Rib6LA'),(89018815,'Rib4LA'),(89018618,'Rib2LA'),(89018420,'Rib1LA'),
                (89534241, 'Rib10RA'),(89534350, 'Rib8RA'), (89519613, 'Rib6RA'),(89518815,'Rib4RA'),(89518618,'Rib2RA'),(89518420,'Rib1RA'),
                (88178168, 'Head C.G.'),(88178675, 'Glabella'),(88176493, 'Head Top'),(87000751,'C1Am'),
                (88265606,'eyeLU'),(88263462,'eyeRU'),(88175601,'eyeLL'),(88170878,'eyeRL'),(88261629,'nose'),
                (88143570,''),(88261630,''),(88135352,''),
                (88175279,''),(88170516,''),(88167682, 'chin'),
                (82008321,''),(82007994,''),(82008524,''),(82006458,''),(82008529,''),(82007910,'',),
                (81008321,''),(81007994,''),(81008524,''),(81006458,''),(81008529,''),(81007910,'',)
                ],
                'Xpositive': [(89063788, 'T12Pm'),(89000726,'T1Pm'),
                (89053454,'Rib1LP'),(89004188,'Rib3LP'),(89019330,'Rib5LP'),(89004321,'Rib7LP'),(89004461,'Rib9LP'),(89004320,'Rib11LP'),
                (89553454,'Rib1RP'),(89504188,'Rib3RP'),(89519330,'Rib5RP'),(89504321,'Rib7RP'),(89504461,'Rib9RP'),(89504320,'Rib11RP'),
                (87000262,'C1Pm'),
                (89006076,'Clavicle-Li'),(89009078,'Clavicle-Lo'),(89506076,'Clavicle-Ri'),(89509078,'Clavicle-Ro'),
                ],
                'Ynegative':[(88175941,'gonionL'),],
                'Ypositive':[(88171241,'gonionR')],
            },
            "transform_sets": ["100"],
            "source_shell_nodes": ["83200101", "83700101","89200801","89700801","88000222", "88000230","88000229", "88000231","88000221","88000223","87200101", "87700101","89200701", "89700701",
                                "82200001","82200401","82200601","82201101","82201301","81200001","81200401","81200601","81201101","81201301",
                                "89200701","86200001","86200301","86200501","86200801","86201001","89700701","85200001","85200301","85200501","85200801","85201001",
                                ],
            "total_side_num": 1000,
            "total_plane_num": 100,
            "modify_nodes": [(83010467, "X", 50),(89066294, "X", 50), (89066279, "X", 50),(83012513, "X", 50),(83512514, "X", 50),
                            (89063788, "Z", -20),(89063788, "X", -20),
                            (88261629, "X", -10),(88167682,"Z",-15),(88175941,"Z",-15),(88171241,"Z",-15),
                            (82008524,"Y",-4),(81008524,"Y",4),(82008529,"Y",-1),(81008529,"Y",1)
                            ]     
        },
    ]

    task2_settings_list = [
        {#左腿
            "inner_method": "set",  # 获取内部节点的方法
            "inner_param": ["201"],  # 获取内部节点的参数
            "source_shell_method": "pid",  # 获取外壳节点的方法
            "source_shell_param": ["82200001", "82200401", "82200601", "82201101", "82201301"],  # 外壳节点
            "source_bone_method": "set",  # 获取骨骼节点的方法
            "source_bone_param": ["108"],  # 骨骼节点
            "shell_total_num": 1000,  # 外壳节点数量
            "bone_total_num": 1000,# 骨骼节点数量
            'rules_to_run':["全腿规则"]
        },
        {#左手
            "inner_method": "set",  # 获取内部节点的方法
            "inner_param": ["204"],  # 获取内部节点的参数
            "source_shell_method": "pid",  # 获取外壳节点的方法
            "source_shell_param": ["89200701","86200001","86201001","86200301","86200501","86200801","86201001"],  # 外壳节点
            "source_bone_method": "set",  # 获取骨骼节点的方法
            "source_bone_param": ["111"],  # 骨骼节点
            "shell_total_num": 1000,  # 外壳节点数量
            "bone_total_num": 1000,  # 骨骼节点数量
            'rules_to_run':["全手部规则", "全肩胸规则",]
        },
        {#头
            "inner_method": "set",  # 获取内部节点的方法
            "inner_param": ["202"],  # 获取内部节点的参数
            "source_shell_method": "pid",  # 获取外壳节点的方法
            "source_shell_param": ["88000222", "88000230","88000229", "88000231","88000221","88000223","87200101", "87700101"],  # 外壳节点
            "source_bone_method": "set",  # 获取骨骼节点的方法
            "source_bone_param": ["109"],  # 骨骼节点
            "shell_total_num": 1000,  # 外壳节点数量
            "bone_total_num": 1000  # 骨骼节点数量
        },
        {#臀
            "inner_method": "set",  # 获取内部节点的方法
            "inner_param": ["205"],  # 获取内部节点的参数
            "source_shell_method": "pid",  # 获取外壳节点的方法
            "source_shell_param": ["83200101", "83700101"],  # 外壳节点
            "source_bone_method": "set",  # 获取骨骼节点的方法
            "source_bone_param": ["112"],  # 骨骼节点
            "shell_total_num": 1000,  # 外壳节点数量
            "bone_total_num": 1000  # 骨骼节点数量
        },
        {#胸
            "inner_method": "set",  # 获取内部节点的方法
            "inner_param": ["203"],  # 获取内部节点的参数
            "source_shell_method": "pid",  # 获取外壳节点的方法
            "source_shell_param": ["89200801", "89700801"],  # 外壳节点
            "source_bone_method": "set",  # 获取骨骼节点的方法
            "source_bone_param": ["110"],  # 骨骼节点
            "shell_total_num": 1000,  # 外壳节点数量
            "bone_total_num": 1000  # 骨骼节点数量
        },
    ]
    # 运行所有任务
    main(task_settings_list, task2_settings_list)