import os
# 设置环境变量来避免内存泄漏
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from utils_adjust import *
from utils_node import *
from utils_reflect import *

def rbf_transform_3d_chunked(all_points, source_control_points, target_control_points, alpha, chunk_size=20000,kernel=None, **kwargs):
    """
    使用 RBF 插值对目标点进行变换，支持节点 ID 并分块计算。默认使用薄板样条径向基函数
    
    参数:
        all_points (np.array): 包含 `node_id` 和坐标的数组 (N x 4)，每行 [node_id, x, y, z]。
        source_control_points (np.array): 包含 `node_id` 和坐标的源控制点数组 (M x 4)。
        target_control_points (np.array): 包含 `node_id` 和坐标的目标控制点数组 (M x 4)。
        alpha (float): 正则化参数，用于稳定计算。
        chunk_size (int): 每次处理的目标点数量。
        kernel (str, optional): 径向基函数名称，可选 'linear'、'cubic'、'gaussian'、'multiquadric'，默认 None 表示薄板样条。
        kwargs: 其他需要传递给核函数的参数（如 epsilon）。
        #kernel='gaussian',  # 选择高斯核
        #epsilon=0.5  # 高斯核的形状参数
        
    返回:
        transformed_points (np.array): 包含变换后坐标和 `node_id` 的数组 (N x 4)，每行 [node_id, x', y', z']。
    """
    def rbf_phi(r, kernel=None, **kwargs):
        """
        根据核函数类型选择径向基函数。
        
        参数:
            r (np.array): 距离矩阵。
            kernel (str): 核函数类型，可选 None 表示默认使用薄板样条。
            kwargs: 额外参数。
        返回:
            np.array: 径向基函数的值。
        """
        epsilon = kwargs.get('epsilon', 1.0)  # 默认参数 epsilon
        if kernel is None or kernel == 'thin_plate_spline':
            # 默认使用薄板样条
            return r**2 * np.log(r + 1e-8)  # 添加 1e-8 防止 log(0)
        elif kernel == 'linear':
            return r  
        elif kernel == 'cubic':
            return r**3
        elif kernel == 'gaussian':
            return np.exp(-(epsilon * r)**2)
        elif kernel == 'multiquadric':
            return np.sqrt(1 + (epsilon * r)**2)
        else:
            raise ValueError(f"未知的核函数类型: {kernel}")

    # 提取坐标部分，忽略 `node_id`
    all_coords = all_points[:, 1:]  # (N x 3)
    source_control_coords = source_control_points[:, 1:]  # (M x 3)
    target_control_coords = target_control_points[:, 1:]  # (M x 3)

    # Step 1: 计算 RBF 权重和线性参数
    N = len(source_control_coords)
    dist_matrix = np.sqrt(((source_control_coords[:, None, :] - source_control_coords[None, :, :])**2).sum(axis=2))
    Phi = rbf_phi(dist_matrix) + alpha * np.identity(N)  # 添加正则化项
    P = np.column_stack([np.ones(N), source_control_coords])  # 线性约束矩阵
    M = np.block([[Phi, P], [P.T, np.zeros((P.shape[1], P.shape[1]))]])  # 插值矩阵
    D = np.vstack([target_control_coords, np.zeros((P.shape[1], target_control_coords.shape[1]))])  # 位移矩阵
    W = np.linalg.solve(M, D)  # 求解线性方程组
    weights, linear_params = W[:-P.shape[1]], W[-P.shape[1]:]

    # Step 2: 分块计算目标点
    transformed_coords = np.empty_like(all_coords)
    for start in range(0, len(all_coords), chunk_size):
        end = min(start + chunk_size, len(all_coords))
        chunk = all_coords[start:end]

        # 计算当前块的距离矩阵
        dist_matrix = np.linalg.norm(chunk[:, None, :] - source_control_coords[None, :, :], axis=2)
        Phi_chunk = rbf_phi(dist_matrix)
        P_chunk = np.column_stack([np.ones(len(chunk)), chunk])

        # 插值变换
        transformed_coords[start:end] = Phi_chunk @ weights + P_chunk @ linear_params

    # Step 3: 合并 `node_id` 和变换后的坐标
    transformed_points = np.column_stack([all_points[:, 0], transformed_coords])
    
    return transformed_points


def compute_projection_points(all_nodes, target_all_nodes, alpha_selected):
    """
    根据初次变换后的节点集合，通过投影计算目标控制点。
    
    参数:
        all_nodes (list): 初次变换后的节点实体列表，每个节点包含 `_id` 和 `position` 属性。
        target_all_nodes (list): 目标网格的节点实体列表，每个节点包含 `_id` 和 `position` 属性。
        alpha_selected (float): 平滑参数，用于调整距离的权重。
    
    返回:
        projected_points (np.array): 投影生成的目标控制点 (N x 4)，每行包含 [node_id, x', y', z']。
    """
    # 提取坐标和 ID
    all_points = np.array([[node._id, *node.position] for node in all_nodes])
    target_points = np.array([[node._id, *node.position] for node in target_all_nodes])

    coords = all_points[:, 1:]  # 初次变换后的点坐标 (N x 3)
    target_coords = target_points[:, 1:]  # 目标网格点坐标 (M x 3)

    # 构建 KD-Tree
    kdtree = KDTree(target_coords)

    # 对每个点找到目标网格中最近的 3 个点
    distances, indices = kdtree.query(coords, k=3)

    projected_coords = []
    projected_ids = []
    for i, (dists, inds) in enumerate(zip(distances, indices)):
        # 获取最近邻目标点的坐标和 ID
        neighbor_coords = target_coords[inds]
        neighbor_ids = target_points[inds, 0]  # 最近邻目标点的 ID
        
        # 根据距离计算权值
        weights = 1 / (dists + alpha_selected)  # 防止 dists 为 0
        normalized_weights = weights / weights.sum()

        # 计算投影点的坐标
        new_coord = (neighbor_coords * normalized_weights[:, None]).sum(axis=0)
        projected_coords.append(new_coord)

        # 选取加权最重的 ID 作为投影点的 ID
        selected_id = neighbor_ids[np.argmax(weights)]
        projected_ids.append(selected_id)

    # 将目标点的 ID 和投影后的坐标组合
    projected_points = np.column_stack([projected_ids, np.array(projected_coords)])

    return projected_points


def select_uniform_nodes(all_nodes, num_control_points):
    """
    使用 K-means 聚类方法根据节点空间分布选择均匀分布的节点，确保没有重复的节点。
    
    参数:
        all_nodes (list): 包含节点实体对象的列表，每个节点对象具有 _id 和 position 属性。
        num_control_points (int): 要选取的节点数量。
    
    返回:
        selected_nodes (list): 均匀选取的节点实体对象列表，确保没有重复。
    """
    total_nodes = len(all_nodes)

    # 如果选取的节点数量大于或等于所有节点数，则返回所有节点
    if num_control_points >= total_nodes:
        print("控制点数量大于或等于节点总数，返回所有节点。")
        return all_nodes

    # 获取所有节点的坐标
    coordinates = np.array([node.position for node in all_nodes])

    # 使用 KMeans 算法进行聚类
    kmeans = KMeans(n_clusters=num_control_points, n_init=10, random_state=42)
    kmeans.fit(coordinates)

    # 获取每个簇的中心点
    cluster_centers = kmeans.cluster_centers_

    # 用于存储选中的节点 ID，确保无重复
    selected_nodes = []
    selected_ids = set()

    # 对每个簇的中心点，选择距离最近的节点
    for center in cluster_centers:
        # 计算每个节点到该中心的距离
        distances = np.linalg.norm(coordinates - center, axis=1)
        
        # 按距离排序，选择最小距离的节点
        sorted_indices = np.argsort(distances)
        
        for idx in sorted_indices:
            node = all_nodes[idx]
            if node._id not in selected_ids:
                # 找到未选中的节点，加入选中节点列表
                selected_nodes.append(node)
                selected_ids.add(node._id)
                break

    return selected_nodes

def select_symmetric_uniform_nodes(all_nodes, total_side_num, total_plane_num):
    """
    根据对称平面计算均匀分布的对称节点，并动态选择规则找到对应的对称点。

    参数：
        - all_nodes (list): 所有节点的实体对象列表，节点具有 _id 和 position 属性。
        - total_side_num (int): 左右两侧控制点的总数量（两侧合计，不含对称面）。
        - total_plane_num (int): 从对称面上选择的控制点数量。

    返回：
        - selected_symmetric_nodes (list): 对称点列表，均匀选取后的节点实体对象（左右和对称面合并）。
    """
    # 获取对称平面并分离节点
    REFLECTION_PLANE,left_nodes, right_nodes, plane_nodes = get_reflection_plane_and_separate_nodes(all_nodes)

    # 从左侧节点中均匀选点
    left_side_num = total_side_num // 2
    selected_left_nodes = select_uniform_nodes(left_nodes, left_side_num)
    #print(f"选中左侧点 ID: {[node._id for node in selected_left_nodes]}")

    # 调用 dynamic_id_replace_rule 获取对称 ID
    symmetric_ids = dynamic_id_replace_rule(selected_left_nodes)

    # 查找对称节点
    selected_right_nodes = []
    for node, symmetric_id in zip(selected_left_nodes, symmetric_ids):
        if symmetric_id:
            symmetric_node = next((n for n in all_nodes if n._id == symmetric_id), None)
            if symmetric_node:
                selected_right_nodes.append(symmetric_node)
            else:
                print(f"对称节点 ID: {symmetric_id} 不存在于 all_nodes 中")
        else:
            print(f"无法为节点 ID {node._id} 生成对称 ID")

    # 从对称面上的节点中均匀选点
    selected_plane_nodes = select_uniform_nodes(plane_nodes, total_plane_num)

    # # 输出调试信息
    # print(f"选中右侧对称点 ID: {[node._id for node in selected_right_nodes]}")
    # print(f"选中左侧对称点 ID: {[node._id for node in selected_left_nodes]}")
    # print(f"选中对称面点 ID: {[node._id for node in selected_plane_nodes]}")

    # 合并左右点和对称面点
    selected_symmetric_nodes = selected_right_nodes + selected_left_nodes + selected_plane_nodes

    # 返回最终结果（左右两侧点和对称面点合并）
    return selected_symmetric_nodes
