import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree
# def enforce_coordinate_uniformity(set_nodes, axis='y',left_nodes=None, smoothing_factor=0.1):
#     """
#     对齐 set_nodes 的指定轴坐标到均值，并在指定条件下调整 left_nodes 的坐标。

#     参数：
#     - set_nodes: 用于构建对称面的节点对象列表，包含 `.position` 属性 (tuple: (x, y, z))。
#     - left_nodes: 左侧节点列表，所有点需满足指定轴坐标小于对称面的值。默认为 None，不进行调整。
#     - axis: 字符串，指定要对齐的轴，支持 'x'、'y'、'z'，默认值为 'y'。
#     - smoothing_factor: 平滑因子，用于动态计算偏移值比例（默认值为 0.1）。

#     返回：
#     - 无直接返回值，直接修改 `set_nodes` 和（如果适用）`left_nodes` 的 `.position` 属性。
#     """
#     axis_map = {'x': 0, 'y': 1, 'z': 2}
#     if axis not in axis_map:
#         raise ValueError(f"Invalid axis '{axis}'. Valid options are 'x', 'y', or 'z'.")

#     axis_idx = axis_map[axis]

#     # Step 1: 对齐 set_nodes 的指定轴到均值
#     coords = np.array([node.position for node in set_nodes])
#     mean_value = np.mean(coords[:, axis_idx])

#     print(f"对齐 set_nodes 的 {axis.upper()} 坐标，设置为均值: {mean_value:.6f}")
#     for node in set_nodes:
#         pos = list(node.position)
#         pos[axis_idx] = mean_value
#         node.position = tuple(pos)

#     if left_nodes is None:
#         print("left_nodes 为 None，跳过调整步骤，仅对齐 set_nodes。")
#         return

#     # 动态计算 offset
#     axis_range = np.ptp(coords[:, axis_idx])  # 计算 set_nodes 的轴范围（最大值 - 最小值）
#     offset = -smoothing_factor * axis_range
#     print(f"动态计算 offset 为: {offset:.6f}")

#     # Step 2: 调整 left_nodes 的坐标
#     print(f"对称面点的 {axis.upper()} 坐标均值为: {mean_value:.6f}")

#     # 找到需要调整的点
#     nodes_to_adjust = [node for node in left_nodes if node.position[axis_idx] > mean_value]
#     print(f"找到 {len(nodes_to_adjust)} 个需要调整的点，其 ID 为:")

#     for node in nodes_to_adjust:
#         pos = list(node.position)
#         original_value = pos[axis_idx]

#         # 调整值，确保小于均值
#         pos[axis_idx] = mean_value + offset
#         node.position = tuple(pos)

#         print(f"点 ID: {node._id}, 原始值: {original_value:.6f} -> 新值: {node.position[axis_idx]:.6f}")

import numpy as np
from scipy.interpolate import splprep, splev

def enforce_coordinate_uniformity(set_nodes, axis='y', left_nodes=None, smoothing_factor=0.1):
    """
    对齐 set_nodes 的指定轴坐标到均值，并在指定条件下调整 left_nodes 的坐标。
    仅在 `axis == 'o'` 时对 `set_nodes` 进行平滑处理，不处理 `left_nodes`。
    """

    def sort_nodes_by_proximity(nodes):
        """
        按节点的空间距离排序，确保顺序符合几何分布。
        """
        if not nodes:
            return []

        sorted_nodes = [nodes[0]]
        remaining_nodes = nodes[1:]

        while remaining_nodes:
            current_node = sorted_nodes[-1]
            current_pos = np.array(current_node.position)
            distances = [np.linalg.norm(np.array(node.position) - current_pos) for node in remaining_nodes]
            nearest_idx = np.argmin(distances)
            sorted_nodes.append(remaining_nodes.pop(nearest_idx))

        return sorted_nodes
    
    if axis == 'o':
        # 仅平滑处理 `set_nodes`，不处理 `left_nodes`
        if set_nodes:
            sorted_set_nodes = sort_nodes_by_proximity(set_nodes)
            coords = np.array([node.position for node in sorted_set_nodes])
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

            # 使用三次样条平滑曲线
            tck, _ = splprep([x, y, z], s=len(sorted_set_nodes) * 10, k=3, per=True)
            fitted_coords = splev(np.linspace(0, 1, len(sorted_set_nodes)), tck)

            # 更新 `set_nodes` 的位置
            for i, node in enumerate(sorted_set_nodes):
                node.position = tuple(fitted_coords[j][i] for j in range(3))
                print(f"更新 set_nodes 点 ID: {node._id}, 新位置: {node.position}")
        else:
            print("`set_nodes` 列表为空，跳过平滑处理。")

    else:
        # 处理普通轴 ('x', 'y', 'z') 的情况
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        if axis not in axis_map:
            raise ValueError(f"无效的轴 '{axis}'，有效选项为 'x', 'y', 'z' 或 'o'。")
        
        axis_idx = axis_map[axis]

        # 对齐 `set_nodes` 的指定轴到均值
        coords = np.array([node.position for node in set_nodes])
        mean_value = np.mean(coords[:, axis_idx])
        print(f"对齐 `set_nodes` 的 {axis.upper()} 轴坐标至均值: {mean_value:.6f}")

        for node in set_nodes:
            pos = list(node.position)
            pos[axis_idx] = mean_value
            node.position = tuple(pos)

        # 如果 `left_nodes` 为空，直接返回
        if not left_nodes:
            print("`left_nodes` 为 None，跳过调整步骤，仅对齐 `set_nodes`。")
            return

        # 动态计算偏移值
        axis_range = np.ptp(coords[:, axis_idx])  # `set_nodes` 的轴范围
        offset = -smoothing_factor * axis_range
        print(f"动态计算偏移值: {offset:.6f}")

        # 调整 `left_nodes` 的坐标
        nodes_to_adjust = [node for node in left_nodes if node.position[axis_idx] > mean_value]
        print(f"找到需要调整的 {len(nodes_to_adjust)} 个点，其 ID 为:")

        for node in nodes_to_adjust:
            pos = list(node.position)
            original_value = pos[axis_idx]
            pos[axis_idx] = mean_value + offset  # 确保调整值小于均值
            node.position = tuple(pos)
            print(f"点 ID: {node._id}, 原始值: {original_value:.6f} -> 新值: {node.position[axis_idx]:.6f}")

def calculate_dynamic_reflection_plane(set_nodes):
    """
    动态计算反射平面法向量和一个平面点。
    在计算前，强制检查和修正平面点的 Y 坐标。

    参数：
    - set_nodes: 节点对象列表，至少包含 3 个节点，节点需有 `.position` 属性。

    返回：
    - normal: NumPy 数组，表示平面的归一化法向量。
    - point_on_plane: NumPy 数组，表示平面上的一个点。
    """
    enforce_coordinate_uniformity(set_nodes,'y')  # 确保平面点 Y 坐标一致
    coords = np.array([node.position for node in set_nodes])
    if len(coords) < 3:
        raise ValueError("定义平面需要至少 3 个点！")
    p1, p2, p3 = coords[:3]
    vector1 = p2 - p1
    vector2 = p3 - p1
    normal = np.cross(vector1, vector2)
    normal = normal / np.linalg.norm(normal)  # 归一化
    point_on_plane = p1  # 平面上的一个点
    print(f"动态计算法向量: {normal}, 平面上的点: {point_on_plane}")
    return normal, point_on_plane

def reflect_coordinates_with_near_plane_handling(coords, normal, point_on_plane, tolerance=1e-6):
    """
    根据反射平面计算节点坐标的镜像。如果节点非常靠近平面，则直接返回原始坐标。

    参数：
    - coords: NumPy 数组，形状为 (N, 3)，表示源节点坐标。
    - normal: NumPy 数组，表示反射平面的法向量。
    - point_on_plane: NumPy 数组，表示反射平面上的一个点。
    - tolerance: 浮点数，表示点到平面的容差，默认值为 1e-6。

    返回：
    - reflected_coords: NumPy 数组，形状为 (N, 3)，表示对称后的节点坐标。
    """
    reflected_coords = []
    for coord in coords:
        vector_to_plane = coord - point_on_plane
        distance_to_plane = np.dot(vector_to_plane, normal)
        if abs(distance_to_plane) < tolerance:  # 靠近平面的点直接镜像
            reflected_coords.append(coord)
        else:
            reflected_coord = coord - 2 * distance_to_plane * normal
            reflected_coords.append(np.round(reflected_coord, decimals=6))
    return np.array(reflected_coords)