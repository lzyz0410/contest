import numpy as np
def enforce_coordinate_uniformity(set_nodes, axis='y'):
    """
    强制平面点在指定轴上的坐标相同。
    如果指定轴的坐标不同，取均值并将所有点的该轴坐标设置为均值。

    参数：
    - set_nodes: 节点对象列表，包含 `.position` 属性 (tuple: (x, y, z))。
    - axis: 字符串，指定要统一的轴，支持 'x'、'y'、'z'，默认值为 'y'。

    返回：
    - 无直接返回值，直接修改 `set_nodes` 的 `.position` 属性。
    """
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if axis not in axis_map:
        raise ValueError(f"Invalid axis '{axis}'. Valid options are 'x', 'y', or 'z'.")

    axis_idx = axis_map[axis]
    coords = np.array([node.position for node in set_nodes])
    axis_values = coords[:, axis_idx]

    if not np.allclose(axis_values, axis_values[0], atol=1e-6):
        average_value = np.mean(axis_values)
        print(f"{axis.upper()} 坐标不一致，强制设置为均值: {average_value:.6f}")
        for node in set_nodes:
            pos = list(node.position)
            pos[axis_idx] = average_value
            node.position = tuple(pos)
    else:
        print(f"所有平面点的 {axis.upper()} 坐标已一致，无需调整。")

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