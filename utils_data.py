import os
import time
import pandas as pd
import numpy as np
import re
import sys

# 用于缓存 *NODE 和 *ELEMENT_SHELL 块的起始和结束位置
block_indices = {
    "*NODE": {"start": None, "end": None},
    "*ELEMENT_SHELL": {"start": None, "end": None}
}


def find_block(file_lines, block_name):
    """
    通用函数：找到特定块的起始和结束行号。
    
    参数:
        file_lines (list): 文件内容的每一行。
        block_name (str): 数据块名称，例如 "*NODE" 或 "*ELEMENT_SHELL"。
    
    返回:
        tuple: (start_index, end_index)，起始行号和结束行号。
    """
    if block_name not in block_indices:
        raise ValueError(f"Unsupported block name: {block_name}")

    # 检查缓存
    if block_indices[block_name]["start"] is not None:
        return block_indices[block_name]["start"], block_indices[block_name]["end"]

    # 找到块的起始和结束位置
    start_index, end_index = 0, 0
    for i, line in enumerate(file_lines):
        if block_name in line:  # 找到块的起始位置
            start_index = i + 1
            break

    for i in range(start_index, len(file_lines)):
        if '*' in file_lines[i] and block_name not in file_lines[i]:  # 找到下一个数据块
            end_index = i
            break

    # 缓存结果
    block_indices[block_name]["start"] = start_index
    block_indices[block_name]["end"] = end_index

    return start_index, end_index


def read_node_coordinates(input_file, target_nodeids=None):
    """
    从文件中读取 *NODE 块的所有节点及其坐标，或者根据给定的 nodeid 只读取指定的节点。
    
    参数:
        input_file (str): 输入的 .k/.key/.inc 文件路径。
        target_nodeids (set, optional): 指定的目标节点 ID 集合。如果为 None，则读取所有节点数据。
    
    返回:
        node_data (np.array): 包含节点 ID 和坐标 (X, Y, Z) 的 NumPy 数组。
        file_lines (list): 文件的所有行内容。
    """
    with open(input_file, 'r') as f:
        file_lines = f.readlines()

    print('提取节点数据...')
    
    # 清空缓存，确保每次读取时都使用新的索引
    block_indices["*NODE"]["start"] = None
    block_indices["*NODE"]["end"] = None

    # 获取 *NODE 块的起始和结束位置
    start_index, end_index = find_block(file_lines, "*NODE")

    if target_nodeids:
        # 使用集合进行查找，直接比较整数 ID，提高效率
        node_data = [
            [line.split()[0], *map(float, line.split()[1:4])]  # 提取节点 ID 和坐标
            for line in file_lines[start_index:end_index]
            if line.strip() and not line.startswith('$')  # 跳过空行和注释行
            and int(line.split()[0]) in target_nodeids  # 直接使用整数进行比较
        ]
    else:
        # 如果没有提供 target_nodeids，读取所有节点数据
        node_data = [
            [line.split()[0], *map(float, line.split()[1:4])]  # 提取节点 ID 和坐标
            for line in file_lines[start_index:end_index]
            if line.strip() and not line.startswith('$')  # 跳过空行和注释行
        ]

    print(f'提取到 {len(node_data)} 个节点.')
    return np.array(node_data, dtype=object), file_lines



def get_node_ids_by_pid(input_file, pids):
    """
    根据多个 PID 获取其关联的唯一节点 ID 和统计信息。

    参数:
        input_file (str): 输入文件路径。
        pids (list): 目标 PID 列表。

    返回:
        tuple: 
            - pid_node_count (dict): 每个 PID 的唯一节点数量。
            - total_node_count (int): 所有目标 PID 的总唯一节点数。
            - pid_to_nodes (dict): 每个 PID 对应的节点 ID 集合。
    """
    with open(input_file, 'r') as f:
        file_lines = f.readlines()

    start, end = find_block(file_lines, "*ELEMENT_SHELL")
    pid_to_nodes = {pid: set() for pid in pids}  # 初始化每个 PID 的节点集合

    for line in file_lines[start:end]:
        if line.strip() and not line.startswith('$'):  # 跳过空行和注释行
            pid = line[8:16].strip()  # 提取 PID
            node_ids = [line[i:i + 8].strip() for i in range(16, len(line), 8) if line[i:i + 8].strip()]
            if pid in pid_to_nodes:
                pid_to_nodes[pid].update(node_ids)

    # 最后去除空节点并统计
    for pid, nodes in pid_to_nodes.items():
        pid_to_nodes[pid] = {nid for nid in nodes if nid}  # 去除空节点

    total_node_count = len(set.union(*pid_to_nodes.values()))
    pid_node_count = {pid: len(nodes) for pid, nodes in pid_to_nodes.items()}

    return pid_node_count, total_node_count, pid_to_nodes


def export_pid_node_ids_to_csv(pids, pid_to_nodes, output_csv_path):
    """
    将每个 PID 对应的节点 ID 导出为 CSV 文件。方便验证结果，并包含 PID_Name 和 Node_ID 列头。

    参数:
        pids (list): 目标 PID 列表。
        pid_to_nodes (dict): 每个 PID 对应的节点 ID 集合。
        output_csv_path (str): 输出 CSV 文件路径。
    """
    # 创建数据字典，按 PID 构造列
    data = {pid: list(pid_to_nodes.get(pid, [])) for pid in pids}
    max_length = max(len(nodes) for nodes in data.values())

    # 对齐所有列的长度，填充 None
    for pid in data:
        data[pid] += [None] * (max_length - len(data[pid]))

    # 转换为 DataFrame
    df = pd.DataFrame(data)

    # 创建与行数一致的第一列内容
    pid_column = ["Node_ID"] + [None] * (max_length - 1)  # 第一行为 "Node_ID"，其他行为 None
    df.insert(0, "PID_Name", pid_column)  # 插入 PID_Name 列

    # 导出为 CSV 文件
    df.to_csv(output_csv_path, index=False, header=True)


def read_csv_by_two_columns(file_path, range_str):
    """
    从 CSV 文件中按指定的字母范围 (如 'A2:B35') 读取数据。

    参数:
        file_path (str): CSV 文件路径。
        range_str (str): 数据范围字符串，例如 'A2:B35'。

    返回:
        list: 返回 (名称, PID) 的元组列表，例如 [("Name1", "81200601"), ("Name2", "81200401")].
    """
    # 加载 CSV 文件
    df = pd.read_csv(file_path)

    # 解析范围字符串
    start_row = int(range_str.split(":")[0][1:]) - 1  # 起始行（转换为 0 索引）
    end_row = int(range_str.split(":")[1][1:]) - 1  # 结束行（转换为 0 索引）

    # 确定列名
    col_letter_start = range_str.split(":")[0][0]  # 起始列字母
    col_letter_end = range_str.split(":")[1][0]    # 结束列字母

    start_col = df.columns[ord(col_letter_start.upper()) - ord('A')]
    end_col = df.columns[ord(col_letter_end.upper()) - ord('A')]

    # 提取范围的数据
    result = df.loc[start_row-1:end_row, [start_col, end_col]]

    # 返回 (名称, PID) 的元组列表
    return list(result.itertuples(index=False, name=None))

def read_csv_single_column(file_path, range_str):
    """
    从 CSV 文件中按指定的单列范围 (如 'B2:B35' 或 'C2:C') 读取一列数据。

    参数:
        file_path (str): CSV 文件路径。
        range_str (str): 数据范围字符串，例如 'C2:C35' 或 'C2:C'。

    返回:
        list: 返回单列数据的列表，例如 ["81200601", "81200401", "81200001"]。
    """
    # 加载 CSV 文件
    df = pd.read_csv(file_path)

    # 解析范围字符串
    col_letter = range_str.split(":")[0][0]  # 提取列字母
    start_row = int(range_str.split(":")[0][1:]) - 1  # 起始行（转换为 0 索引）
    
    # 检查结束部分是否带有数字
    end_part = range_str.split(":")[1]
    if len(end_part) > 1 and end_part[1:].isdigit():
        end_row = int(end_part[1:]) - 1  # 如果有数字，则按数字处理
    else:
        # 没有数字，则提取到该列的最后一行
        column_name = df.columns[ord(col_letter.upper()) - ord('A')]
        end_row = df[column_name].last_valid_index()

    # 确定列名
    column_name = df.columns[ord(col_letter.upper()) - ord('A')]

    # 提取指定范围的数据
    result = df.loc[start_row-1:end_row, column_name]

    # 转换为字符串列表并去除多余空格
    return result.astype(str).str.strip().tolist()


def write_modified_coordinates(output_file, file_lines, updated_node_data):
    """
    将修改后的节点坐标写回文件，并显示写入进度条。
    参数:
        output_file (str): 输出文件路径。
        file_lines (list): 原始文件的所有行内容。
        updated_node_data (np.ndarray): 包含每个节点的数据，是 numpy.ndarray 数组。
    """
    
    # 清空缓存，确保每次读取时都使用新的索引
    block_indices["*NODE"]["start"] = None
    block_indices["*NODE"]["end"] = None

    # 查找 *NODE 块的位置
    start_index, end_index = find_block(file_lines, "*NODE")
    
    print('开始写入修改后的文件...')

    
    # 转换为字典，方便查找
    id_to_data = {str(int(row[0])): row[1:] for row in updated_node_data}
    
    total_nodes = end_index - start_index  # 总节点数

    # 遍历文件中的节点块并更新坐标
    for i in range(start_index, end_index):
        line = file_lines[i]
        if line.strip() and not line.startswith('$'):  # 跳过注释和空行
            node_id = line.split()[0].strip()
            if node_id in id_to_data:
                x, y, z = id_to_data[node_id]
                # 更新节点坐标
                file_lines[i] = f"{node_id:8} {x:15.6f} {y:15.6f} {z:15.6f}\n"

        # 显示进度条
        if (i - start_index) % (total_nodes // 100 or 1) == 0:
            percent_complete = (i - start_index) / total_nodes * 100
            print(f"\r写入进度: [{'#' * int(percent_complete // 2):<50}] {percent_complete:.1f}%", end='', flush=True)

    # 保存修改后的文件
    with open(output_file, 'w') as f:
        f.writelines(file_lines)
    print("\n写入完成！修改后的文件已保存为:", output_file)



