import os
# 设置环境变量来避免内存泄漏
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import KMeans

def rbf_transform_3d_chunked(all_points, source_control_points, target_control_points, alpha, chunk_size=20000):
    """
    使用 RBF 插值对目标点进行变换，支持节点 ID 并分块计算。
    
    参数:
        all_points (np.array): 包含 `node_id` 和坐标的数组 (N x 4)，每行 [node_id, x, y, z]。
        source_control_points (np.array): 包含 `node_id` 和坐标的源控制点数组 (M x 4)。
        target_control_points (np.array): 包含 `node_id` 和坐标的目标控制点数组 (M x 4)。
        alpha (float): 正则化参数，用于稳定计算。
        chunk_size (int): 每次处理的目标点数量。
        
    返回:
        transformed_points (np.array): 包含变换后坐标和 `node_id` 的数组 (N x 4)，每行 [node_id, x', y', z']。
    """
    def rbf_phi(r):
        """
        基于薄板样条函数定义径向基函数。
        参数:
            r (float): 距离。
        返回:
            (float): RBF 值。
        """
        return r**2 * np.log(r + 1e-8)  # 添加 1e-8 防止 log(0)

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


def compute_projection_points(all_points, target_points, alpha_second):
    """
    根据初次变换后的点集，根据选定的大量点，通过投影计算目标控制点。
    
    参数:
        all_points (np.array): 初次变换后的点集 (N x 4)，每行包含 [node_id, x, y, z]。
        target_points (np.array): 目标网格的所有点 (M x 4)，每行包含 [node_id, x, y, z]。
        alpha_second (float): 平滑参数，用于调整 RBF 插值的正则化。
    
    返回:
        projected_points (np.array): 投影生成的目标控制点 (N x 4)，每行包含 [node_id, x', y', z']。
    """
    # 提取坐标部分，忽略 `node_id`
    coords = all_points[:, 1:]  # (N x 3)
    target_coords = target_points[:, 1:]  # (M x 3)

    # 构建 KD-Tree
    kdtree = KDTree(target_coords)

    # 对每个点找到目标网格中最近的 3 个点
    distances, indices = kdtree.query(coords, k=3)

    projected_points = []
    for i, (dists, inds) in enumerate(zip(distances, indices)):
        # 获取最近邻目标点的坐标
        neighbor_coords = target_coords[inds]
        
        # 根据距离计算权值
        weights = 1 / (dists + alpha_second)  # 防止 dists 为 0
        normalized_weights = weights / weights.sum()

        # 计算投影点的坐标
        new_coord = (neighbor_coords * normalized_weights[:, None]).sum(axis=0)
        projected_points.append(new_coord)

    return np.array(projected_points)


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