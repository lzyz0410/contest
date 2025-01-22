import os
# 设置环境变量来避免内存泄漏
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from utils_adjust import *
from utils_node import *

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

def select_symmetric_uniform_nodes(all_nodes, num_control_points, mapping_file="node_mapping.csv"):
    """
    根据对称平面计算均匀分布的对称节点，并动态选择规则找到对应的对称点。

    参数：
        - all_nodes (list): 所有节点的实体对象列表，节点具有 _id 和 position 属性。
        - num_control_points (int): 要选择的均匀对称点的总数量（包括左右两边）。
        - mapping_file (str): 映射文件路径，默认为 "node_mapping.csv"。

    返回：
        - selected_symmetric_nodes (list): 对称点列表，均匀选取后的节点实体对象（左右合并）。
    """

    def load_mapping_file(mapping_file):
        try:
            df = pd.read_csv(mapping_file)
            if "Left_Node_ID" in df.columns and "Right_Node_ID" in df.columns:
                print(f"从文件 {mapping_file} 成功加载节点映射！")
                # 强制将映射文件中的 ID 转为整数
                df["Left_Node_ID"] = df["Left_Node_ID"].astype(int)
                df["Right_Node_ID"] = df["Right_Node_ID"].astype(int)
                # 注意这里的左右关系调整
                mapping = dict(zip(df["Right_Node_ID"], df["Left_Node_ID"]))
                print(f"映射文件内容（部分预览）: {list(mapping.items())[:10]}")
                return mapping
            else:
                raise ValueError("映射文件缺少必要的列：Left_Node_ID 或 Right_Node_ID")
        except Exception as e:
            print(f"映射文件加载失败: {e}")
            return {}

    # 缓存映射数据
    mapping = load_mapping_file(mapping_file)
    print(f"映射文件内容预览: {list(mapping.items())[:10]}")  # 调试映射数据

    # 动态选择对称规则
    def dynamic_id_replace_rule(node_id):
        try:
            node_id = int(node_id)
            node_id_str = str(node_id)
            if node_id_str.startswith("88"):
                # 使用缓存的映射（基于调整后的左右关系）
                symmetric_id = mapping.get(node_id)
                if symmetric_id is None:
                    print(f"映射文件中未找到源节点 ID: {node_id}")
                return symmetric_id
            elif node_id_str.startswith("83"):
                return int(node_id_str.replace("830", "835", 1))
            elif node_id_str.startswith("81") or node_id_str.startswith("82"):
                return int(node_id_str.replace("81", "82", 1)) if node_id_str.startswith("81") else int(node_id_str.replace("82", "81", 1))
            elif node_id_str.startswith("85") or node_id_str.startswith("86"):
                return int(node_id_str.replace("85", "86", 1)) if node_id_str.startswith("85") else int(node_id_str.replace("86", "85", 1))
            elif node_id_str.startswith("89"):
                return node_id + 500000 if node_id < 89000000 else node_id - 500000
            elif node_id_str.startswith("87"):
                return int(node_id_str.replace("870", "875", 1))
            else:
                print(f"未能为节点 ID {node_id} 匹配规则")
                return None
        except Exception as e:
            print(f"动态规则转换失败，节点 ID: {node_id}, 错误: {e}")
            return None

    # 获取对称平面节点
    plane_nodes = get_all_nodes("set", ["3", "4", "5", "6"])
    enforce_coordinate_uniformity(plane_nodes, axis='y')
    normal, point_on_plane = calculate_dynamic_reflection_plane(plane_nodes)

    # 获取所有节点坐标并分离一侧的节点
    coordinates = np.array([node.position for node in all_nodes])
    right_nodes = []  # 原来的 left_nodes 改为 right_nodes
    for i, coord in enumerate(coordinates):
        source_node = all_nodes[i]
        vector_to_plane = coord - point_on_plane
        side = np.dot(vector_to_plane, normal)
        if side < 0:  # 仅处理右侧节点
            right_nodes.append(source_node)

    # 从右侧节点中均匀选点
    num_right_points = num_control_points // 2  # 原来的 num_left_points 改为 num_right_points
    selected_right_nodes = select_uniform_nodes(right_nodes, num_right_points)  # 改为处理右侧节点

    # 找到右侧选点的对称点
    selected_left_nodes = []  # 原来的 selected_right_nodes 改为 selected_left_nodes
    for node in selected_right_nodes:  # 遍历右侧节点
        symmetric_id = dynamic_id_replace_rule(node._id)
        if symmetric_id is not None:
            symmetric_node = next((n for n in all_nodes if n._id == symmetric_id), None)
            if symmetric_node:
                selected_left_nodes.append(symmetric_node)
            else:
                print(f"对称节点 ID: {symmetric_id} 不存在于 all_nodes 中")
        else:
            print(f"无法为节点 ID {node._id} 生成对称 ID")

    # 输出调试信息
    print(f"选中右侧对称点 ID: {[node._id for node in selected_right_nodes]}")  # 调整输出信息
    print(f"选中左侧对称点 ID: {[node._id for node in selected_left_nodes]}")  # 调整输出信息

    # 合并左右两侧的点
    selected_symmetric_nodes = selected_right_nodes + selected_left_nodes  # 按新的命名合并

    # 返回最终结果（左右两侧点合并）
    return selected_symmetric_nodes
