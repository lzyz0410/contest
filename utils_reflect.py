import numpy as np
import pandas as pd
from utils_node import *
from utils_adjust import *
from collections import defaultdict
from utils_env import *
'''
"腿部规则" "臀部规则" "肩胸规则" "颈部规则" "头部规则" "手部规则"
from reflect_utils import reflect
# 调用 reflect() 函数，执行所有规则
reflect(rules_to_run=True)

# 调用 reflect() 函数，执行指定的规则
reflect(rules_to_run=["手臂规则", "腿部规则"])
'''
# # 缓存变量
# MAPPING_FILE_PATH = "E:\\LZYZ\\Scoliosis\\RBF\\Contest\\final\\node_mapping.csv"
# **调用函数**
MAPPING_FILE_PATH = find_file_in_parents("node_mapping.csv")
mapping = None
reverse_mapping = None
node_classification_cache = {}

def load_mapping_file():
    """
    只加载一次映射文件，返回左->右和右->左的映射字典。
    
    返回：
        - mapping (dict): 左侧节点 ID 到右侧节点 ID 的映射字典。
        - reverse_mapping (dict): 右侧节点 ID 到左侧节点 ID 的映射字典。
    """
    global mapping, reverse_mapping
    if mapping is None and reverse_mapping is None:  # 确保只加载一次
        try:
            df = pd.read_csv(MAPPING_FILE_PATH)
            if "Left_Node_ID" not in df.columns or "Right_Node_ID" not in df.columns:
                raise ValueError("映射文件缺少必要的列：Left_Node_ID 或 Right_Node_ID")
            mapping = dict(zip(df["Left_Node_ID"], df["Right_Node_ID"]))  # 生成左->右的映射关系
            reverse_mapping = {v: k for k, v in mapping.items()}  # 生成右->左的映射关系
            print(f"从文件 {MAPPING_FILE_PATH} 成功加载节点映射！")
        except Exception as e:
            print(f"加载映射文件失败: {e}")

def find_symmetric_node(node_id):
    """ 根据规则返回节点的对称节点 ID """
    node_id_str = str(node_id)

    # 89, 88, 86 等开头的规则都可优化为直接返回结果
    if node_id_str.startswith("89"):
        third_digit = int(node_id_str[2])
        return node_id + 500000 if third_digit <= 4 else node_id - 500000
    if node_id_str.startswith("88") and node_id in mapping:
        return mapping[node_id]
    if node_id_str.startswith("86"):
        return node_id - 1000000
    if node_id_str.startswith("85"):
        return node_id + 1000000
    if node_id_str.startswith("83"):
        third_digit = int(node_id_str[2])
        return node_id + 500000 if third_digit <= 4 else node_id - 500000
    if node_id_str.startswith("87"):
        third_digit = int(node_id_str[2])
        return node_id + 500000 if third_digit <= 4 else node_id - 500000
    if node_id_str.startswith("82"):
        return node_id - 1000000
    if node_id_str.startswith("81"):
        return node_id + 1000000
    return None

def dynamic_id_replace_rule(selected_left_nodes):
    """
    根据节点 ID 的规则生成对称节点的 ID。
    """
    symmetric_ids = [find_symmetric_node(node._id) for node in selected_left_nodes]
    return symmetric_ids

def classify_node(node_id_str, node_id):
    """ 根据节点的 ID 字符串分类节点。 """
    if node_id_str.startswith("88"):  # 88开头的节点，按照映射文件分左右
        return "left" if node_id in mapping else "right" if node_id in reverse_mapping else "plane"
    if node_id_str.startswith(("89", "83", "87")):
        return "left" if int(node_id_str[2]) <= 4 else "right"
    if node_id_str.startswith(("85", "86", "81", "82")):
        return "left" if node_id_str.startswith(("82", "86")) else "right"
    return "plane"

def get_reflection_plane_and_separate_nodes(all_nodes):
    """ 获取对称面平面并将节点分为左右节点和对称面节点 """
    global mapping, reverse_mapping
    if mapping is None or reverse_mapping is None:
        load_mapping_file()  # 确保映射文件被加载

     # 获取对称面节点
    enforce_coordinate_uniformity(get_nodes_from_set([3,4,5,6]), axis='y', left_nodes=None, smoothing_factor=0.1)   
    set56_nodes = get_all_nodes("set", ["3", "4", "5", "6"])
    set56_ids = {node._id for node in set56_nodes}  # 提取节点 ID
    REFLECTION_PLANE = (np.array([0, 1, 0]), np.array(set56_nodes[0].position))  # 反射平面

    # 分离节点
    left_nodes, right_nodes, plane_nodes = [], [], []
    for node in all_nodes:
        node_id = node._id
        node_id_str = str(node_id)

        # 如果节点在对称面上
        if node_id in set56_ids:
            plane_nodes.append(node)
        else:
            # 使用缓存避免重复分类
            if node_id not in node_classification_cache:
                classification = classify_node(node_id_str, node_id)
                node_classification_cache[node_id] = classification
            classification = node_classification_cache[node_id]

            if classification == "left":
                left_nodes.append(node)
            elif classification == "right":
                right_nodes.append(node)
            else:
                plane_nodes.append(node)

    return REFLECTION_PLANE,left_nodes, right_nodes, plane_nodes

def reflect_coordinates_with_near_plane_handling(REFLECTION_PLANE,coords, tolerance=1e-6):
    """
    根据反射平面计算节点坐标的镜像。如果节点非常靠近平面，则直接返回原始坐标。
    
    参数：
    - coords: 需要镜像的坐标数组，形状为 (N, 3)
    - tolerance: 点到平面的容差
    
    返回：
    - reflected_coords: 镜像后的坐标
    """

    normal, point_on_plane = REFLECTION_PLANE
    coords = np.array(coords)  # 确保输入是 NumPy 数组
    vector_to_plane = coords - point_on_plane
    distance_to_plane = np.dot(vector_to_plane, normal)

    # 计算反射的坐标
    reflected_coords = coords - 2 * distance_to_plane[:, np.newaxis] * normal

    # 如果点距离平面非常近，则直接返回原始坐标
    reflected_coords[abs(distance_to_plane) < tolerance] = coords[abs(distance_to_plane) < tolerance]
    
    return np.round(reflected_coords, decimals=6)  # 四舍五入精度到6位

def reflect(rules_to_run=True):
    # **确保 rules_to_run 处理正确**
    normal_rules = ["头部规则", "颈部规则", "肩胸规则", "手部规则", "腿部规则", "臀部规则"]
    full_body_rules = ["全手部规则", "全肩胸规则"]

    if rules_to_run is True:
        rules_to_run = normal_rules  # 运行所有普通规则（不包含全手部、全肩胸）
    elif isinstance(rules_to_run, list):
        if set(rules_to_run).issubset(set(full_body_rules)):  
            rules_to_run = full_body_rules  # 如果只包含全手部、全肩胸，则只运行这些
    else:
        raise ValueError("rules_to_run 必须是 True 或者一个包含规则名称的列表")

    use_set71 = any(rule in rules_to_run for rule in full_body_rules)  # 只有全手部/全肩胸时启用 set71

    # 根据数据源获取节点
    #data_source = ("set", ["71"]) if use_set71 else ("csv", "E:\\LZYZ\\Scoliosis\\RBF\\Contest\\final\\shell_property.csv", "A2:B35")
    # **动态查找 `shell_property.csv`，替换硬编码路径**
    shell_property_path = find_file_in_parents("shell_property.csv")

    # 设置数据源（自动适配路径）
    data_source = ("set", ["71"]) if use_set71 else ("csv", str(shell_property_path), "A2:B35")
    print(f"使用 {'set71' if use_set71 else 'CSV'} 作为数据源")
    all_nodes = get_all_nodes(*data_source)  # 直接解包参数

    # 规则字典（根据数据源动态调整）
    rules = {
        "头部规则": lambda nodes: [node for node in nodes if str(node._id).startswith("88")],
        "颈部规则": lambda nodes: [node for node in nodes if str(node._id).startswith("87")],
        "肩胸规则": lambda nodes: [node for node in nodes if str(node._id).startswith("89")],
        "手部规则": lambda nodes: [node for node in nodes if str(node._id).startswith(("86", "85"))],
        "腿部规则": lambda nodes: [node for node in nodes if str(node._id).startswith(("82", "81"))],
        "臀部规则": lambda nodes: [node for node in nodes if str(node._id).startswith("83")],
    }

    # 若使用 set71，则替换 "肩胸规则" 和 "手部规则" 为 "全肩胸规则" 和 "全手部规则"
    if use_set71:
        rules.update({
            "全肩胸规则": lambda nodes: [node for node in nodes if str(node._id).startswith(("89"))],
            "全手部规则": lambda nodes: [node for node in nodes if str(node._id).startswith(("85", "86"))],
        })

    # 如果未指定 rules_to_run，则执行所有规则
    if rules_to_run is True:
        rules_to_run = list(rules.keys())  # 默认执行所有规则

    # 创建节点 ID 到节点的映射
    node_id_map = {node._id: node for node in all_nodes}

    # 用于存储每个规则的左右节点和对称面节点信息
    rule_summary = {}

    # 用于存储所有规则的总节点数
    total_left_nodes = set()  # 使用集合避免重复
    total_right_nodes = set()
    total_plane_nodes = set()

    # 按规则名称遍历并应用规则
    for rule_name in rules_to_run:
        if rule_name in rules:
            handler = rules[rule_name]  # 获取规则对应的处理函数
            
            # 根据规则筛选需要的节点
            filtered_nodes = handler(all_nodes)  # 从 all_nodes 中筛选符合该规则的节点

            # 获取该规则下的对称平面并分离左右节点
            REFLECTION_PLANE,left_rule_nodes, right_rule_nodes, plane_rule_nodes = get_reflection_plane_and_separate_nodes(filtered_nodes)
            
            # 如果左右节点数量不一致，打印左右节点的 ID
            if len(left_rule_nodes) != len(right_rule_nodes):
                print("警告：左右节点数量不一致！")
                print(f"左侧节点的 ID: {[node._id for node in left_rule_nodes]}")
                print(f"右侧节点的 ID: {[node._id for node in right_rule_nodes]}")

            # 将 left_rule_nodes 转换为 np.array，每行包含 [id, x, y, z]
            left_points = np.array([[node._id] + list(node.position) for node in left_rule_nodes])

            # 确保 left_points 是二维数组
            if left_points.ndim == 1:
                left_points = left_points.reshape(-1, 4)

            # 调用 dynamic_id_replace_rule 获取对称 ID
            symmetric_ids = dynamic_id_replace_rule(left_rule_nodes)

            # 反射变换
            reflected_coords = reflect_coordinates_with_near_plane_handling(REFLECTION_PLANE,left_points[:, 1:],tolerance=1e-6)
            # 更新右侧节点的坐标
            for node, new_coord, symmetric_id in zip(left_rule_nodes, reflected_coords, symmetric_ids):
                if symmetric_id:
                    if symmetric_id in node_id_map:
                        right_node = node_id_map[symmetric_id]
                        right_node.position = tuple(new_coord)  # 更新右侧节点的坐标
                    else:
                        print(f"未找到与节点 ID {node._id} 对应的对称节点 ID {symmetric_id}！")
                else:
                    print(f"无法为节点 ID {node._id} 生成对称 ID")

            # 输出包含规则名称的消息
            print(f"规则 {rule_name} 对称右侧节点坐标更新完成！")

            # 存储当前规则的左右节点和对称面节点信息
            rule_summary[rule_name] = {
                "left_nodes": left_rule_nodes,
                "right_nodes": right_rule_nodes,
                "plane_nodes": plane_rule_nodes,
            }

            # 将当前规则的节点添加到总节点集合中
            total_left_nodes.update(left_rule_nodes)
            total_right_nodes.update(right_rule_nodes)
            total_plane_nodes.update(plane_rule_nodes)

        else:
            print(f"未找到规则: {rule_name}")
    
    # 输出每个规则的左右节点和对称面节点的综合信息
    print("\n所有规则处理完成，综合信息如下：")
    for rule_name, summary in rule_summary.items():
        print(f"规则 {rule_name}:")
        print(f"  左侧节点数量: {len(summary['left_nodes'])}")
        print(f"  右侧节点数量: {len(summary['right_nodes'])}")
        print(f"  对称面节点数量: {len(summary['plane_nodes'])}")
        print("-" * 50)

    # 输出所有规则的总节点数
    print("\n所有规则的总节点数：")
    print(f"  总左侧节点数量: {len(total_left_nodes)}")
    print(f"  总右侧节点数量: {len(total_right_nodes)}")
    print(f"  总对称面节点数量: {len(total_plane_nodes)}")
    print(f"  总节点数量: {len(total_left_nodes) + len(total_right_nodes) + len(total_plane_nodes)}")

    print("所有规则处理完成！") 