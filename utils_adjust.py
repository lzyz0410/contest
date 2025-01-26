import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree


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
                #print(f"更新 set_nodes 点 ID: {node._id}, 新位置: {node.position}")
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
            #print(f"点 ID: {node._id}, 原始值: {original_value:.6f} -> 新值: {node.position[axis_idx]:.6f}")

