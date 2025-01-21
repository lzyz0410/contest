import numpy as np
from utils_node import *
from scipy.spatial import cKDTree

def laplacian_smoothing(pids, boundary_sets=None, iterations=1, alpha=0.5):
    """
    对指定 PID 区域的网格进行拉普拉斯平滑，并结合边界保护和动态权重的控制。
    
    # 用户可配置部分
    pids = ["89200701"]  # 平滑的 PID 区域
    boundary_sets = ["22", "20"]  # 边界保护的 Set ID 列表//=None
    iterations = 10  # 平滑迭代次数
    alpha = 0.5  # 边界节点初始移动权重
    
    参数:
        pids (str or list): 单个 PID（字符串）或多个 PID 的列表，指定需要平滑的区域。
        boundary_sets (list or None): 边界节点的 Set ID 列表，用于构建边界点。如果为 None，则没有边界保护。
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

    ids = list(unique_nodes.keys())
    coords = np.array([node.position for node in unique_nodes.values()])
    nodes = list(unique_nodes.values())
    N = len(coords)
    adjacency = {i: set() for i in range(N)}
    id_to_index = {node_id: idx for idx, node_id in enumerate(ids)}

    # 获取边界点
    if boundary_sets:
        boundary_nodes = get_nodes_from_set(boundary_sets)
        boundary_points = np.array([[node._id] + list(node.position) for node in boundary_nodes])
        boundary_ids = set(boundary_points[:, 0].astype(int))
        boundary_coords = boundary_points[:, 1:]
        boundary_tree = cKDTree(boundary_coords)
    else:
        boundary_ids = set()
        boundary_tree = None

    # 构建邻接表
    for element in elements_info:
        for i, node_id in enumerate(element):
            if node_id in id_to_index:
                for j, neighbor_id in enumerate(element):
                    if i != j and neighbor_id in id_to_index:
                        adjacency[id_to_index[node_id]].add(id_to_index[neighbor_id])

    # 平滑部分
    for iteration in range(iterations):
        new_coords = coords.copy()
        for i in range(N):
            neighbors = list(adjacency[i])
            if len(neighbors) > 0:
                smooth_position = coords[neighbors].mean(axis=0)
                if ids[i] in boundary_ids and boundary_tree:
                    _, dist_to_boundary = boundary_tree.query(coords[i])
                    distance_weight = min(1.0, dist_to_boundary / 5.0)
                    dynamic_weight = alpha / (iteration + 1)
                    weight = distance_weight * dynamic_weight
                    new_coords[i] = weight * smooth_position + (1 - weight) * coords[i]
                else:
                    new_coords[i] = smooth_position
        coords = new_coords

    # 更新 ANSA 中的节点坐标
    smoothed_points = np.column_stack((ids, coords))
    update_ansa_node_coordinates(smoothed_points, nodes)





