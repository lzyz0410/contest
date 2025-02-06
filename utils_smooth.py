import numpy as np
from utils_node import *
from scipy.spatial import cKDTree

def laplacian_smoothing(pids, iterations=1, alpha=0.5):
    """
    对指定 PID 区域的网格进行拉普拉斯平滑，并结合自动边界检测和动态权重的控制。
    
    参数：
        pids (str or list): 单个 PID（字符串）或多个 PID 的列表，指定需要平滑的区域。
        iterations (int): 平滑迭代次数。
        alpha (float): 初始边界节点移动权重因子，范围为 0 到 1。
        
    返回：
        None: 自动更新 ANSA 中的节点坐标。
    """
    if isinstance(pids, str):  # 如果是单个 PID，转换为列表
        pids = [pids]

    # 获取节点和单元信息
    unique_nodes = {}
    elements_info = []
    edge_count = {}

    for pid in pids:
        section_shell = base.GetEntity(constants.LSDYNA, "SECTION_SHELL", int(pid))
        elements = base.CollectEntities(constants.LSDYNA, [section_shell], "ELEMENT_SHELL", recursive=True)
        for element in elements:
            node_values = element.get_entity_values(constants.LSDYNA, ["N1", "N2", "N3", "N4"])
            element_nodes = []
            for node in node_values.values():
                if isinstance(node, base.Entity):
                    if node._id not in unique_nodes:
                        unique_nodes[node._id] = node
                    element_nodes.append(node._id)
            elements_info.append(element_nodes)

            # 统计边的出现次数
            edges = [
                (element_nodes[i], element_nodes[(i + 1) % len(element_nodes)])
                for i in range(len(element_nodes))
            ]
            for edge in edges:
                edge = tuple(sorted(edge))  # 确保边的方向一致
                edge_count[edge] = edge_count.get(edge, 0) + 1

    ids = list(unique_nodes.keys())
    coords = np.array([node.position for node in unique_nodes.values()])
    nodes = list(unique_nodes.values())
    N = len(coords)
    adjacency = {i: set() for i in range(N)}
    id_to_index = {node_id: idx for idx, node_id in enumerate(ids)}

    # 构建邻接表
    for element in elements_info:
        for i, node_id in enumerate(element):
            if node_id in id_to_index:
                for j, neighbor_id in enumerate(element):
                    if i != j and neighbor_id in id_to_index:
                        adjacency[id_to_index[node_id]].add(id_to_index[neighbor_id])

    # 自动检测边界点
    boundary_ids = set()
    for edge, count in edge_count.items():
        if count == 1:  # 如果边只出现一次，说明是边界边
            boundary_ids.update(edge)

    #print(f"检测到的边界点 ID: {sorted(boundary_ids)}")

    # 平滑部分
    for iteration in range(iterations):
        new_coords = coords.copy()
        for i in range(N):
            neighbors = list(adjacency[i])
            if len(neighbors) > 0:
                smooth_position = coords[neighbors].mean(axis=0)
                if ids[i] in boundary_ids:
                    dynamic_weight = alpha / (iteration + 1)
                    new_coords[i] = dynamic_weight * smooth_position + (1 - dynamic_weight) * coords[i]
                else:
                    new_coords[i] = smooth_position
        coords = new_coords

    # 更新 ANSA 中的节点坐标
    smoothed_points = np.column_stack((ids, coords))
    update_ansa_node_coordinates(smoothed_points, nodes)
    print(f'完成{pids}平滑')


import numpy as np
from utils_node import *
from scipy.spatial import cKDTree

def laplacian_smoothing_improved(pids, iterations=10, alpha=0.5, beta=0.3):
    """
    改进型拉普拉斯平滑，使网格适度膨胀，避免向内收缩。

    参数：
        pids (str or list): 需要平滑的 PID（支持单个字符串或列表）。
        iterations (int): 平滑迭代次数，默认 10 次。
        alpha (float): 平滑系数（控制拉普拉斯平滑影响）。
        beta (float): 反向膨胀因子（用于防止网格收缩）。

    返回：
        None: 直接更新 ANSA 节点坐标。
    """
    if isinstance(pids, str):  # 如果是单个 PID，转换为列表
        pids = [pids]

    # 存储唯一节点和单元信息
    unique_nodes = {}
    elements_info = []
    edge_count = {}

    for pid in pids:
        section_shell = base.GetEntity(constants.LSDYNA, "SECTION_SHELL", int(pid))
        elements = base.CollectEntities(constants.LSDYNA, [section_shell], "ELEMENT_SHELL", recursive=True)
        for element in elements:
            node_values = element.get_entity_values(constants.LSDYNA, ["N1", "N2", "N3", "N4"])
            element_nodes = []
            for node in node_values.values():
                if isinstance(node, base.Entity):
                    if node._id not in unique_nodes:
                        unique_nodes[node._id] = node
                    element_nodes.append(node._id)
            elements_info.append(element_nodes)

            # 统计边界信息
            edges = [
                (element_nodes[i], element_nodes[(i + 1) % len(element_nodes)])
                for i in range(len(element_nodes))
            ]
            for edge in edges:
                edge = tuple(sorted(edge))  # 方向一致化
                edge_count[edge] = edge_count.get(edge, 0) + 1

    ids = list(unique_nodes.keys())
    coords = np.array([node.position for node in unique_nodes.values()])
    nodes = list(unique_nodes.values())
    N = len(coords)
    adjacency = {i: set() for i in range(N)}
    id_to_index = {node_id: idx for idx, node_id in enumerate(ids)}

    # 构建邻接表
    for element in elements_info:
        for i, node_id in enumerate(element):
            if node_id in id_to_index:
                for j, neighbor_id in enumerate(element):
                    if i != j and neighbor_id in id_to_index:
                        adjacency[id_to_index[node_id]].add(id_to_index[neighbor_id])

    # **检测边界点**
    boundary_ids = set()
    for edge, count in edge_count.items():
        if count == 1:  # 只出现一次的边界
            boundary_ids.update(edge)

    # **计算法向量**
    normals = np.zeros_like(coords)
    for element in elements_info:
        if len(element) >= 3:
            v1 = coords[id_to_index[element[1]]] - coords[id_to_index[element[0]]]
            v2 = coords[id_to_index[element[2]]] - coords[id_to_index[element[0]]]
            normal = np.cross(v1, v2)
            normal /= np.linalg.norm(normal) + 1e-8
            for node_id in element:
                if node_id in id_to_index:
                    normals[id_to_index[node_id]] += normal
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8  # 归一化

    # **执行迭代平滑**
    for iteration in range(iterations):
        new_coords = coords.copy()
        for i in range(N):
            neighbors = list(adjacency[i])
            if len(neighbors) > 0:
                smooth_position = coords[neighbors].mean(axis=0)  # 计算邻居均值
                displacement = smooth_position - coords[i]  # 计算位移量

                if ids[i] in boundary_ids:
                    dynamic_weight = alpha * 0.5  # 增强边界的平滑度
                    new_coords[i] = (1 - dynamic_weight) * coords[i] + dynamic_weight * smooth_position
                else:
                    expansion = beta * normals[i]  # 沿法向外扩张
                    new_coords[i] = coords[i] + alpha * displacement + expansion

        coords = new_coords  # 更新坐标

    # **更新 ANSA 节点坐标**
    smoothed_points = np.column_stack((ids, coords))
    update_ansa_node_coordinates(smoothed_points, nodes)
    print(f'完成 {pids} 平滑，迭代 {iterations} 次，增强外扩防止收缩。')