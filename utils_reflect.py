import numpy as np
import pandas as pd
from utils_node import *
from utils_adjust import *
'''
"腿部规则" "臀部规则" "胸部规则" "颈部规则" "头部规则" "肩膀规则" "手臂规则"
from reflect_utils import reflect
# 调用 reflect() 函数，执行所有规则
reflect(run_all_rules=True)

# 调用 reflect() 函数，执行指定的规则
reflect(rules_to_run=["手臂规则", "腿部规则"])
'''
# 对称面固定的定义
SET56_METHOD = "set"  # 对称面的获取方法
SET56_IDENTIFIERS = ["3", "4", "5", "6"]  # 对称面的标识符列表

# 全局变量存储对称面的信息
global_reflection_plane = None
def reflect(task_type=None, source_set_method=None, source_set_identifiers=None, target_set_method=None, target_set_identifiers=None, filter_by_prefix=False, mapping_file=None, id_replace_rule=None, run_all_rules=False, rules_to_run=None):
    """
    大函数：处理替换任务。

    参数：
    - task_type: 任务类型，可以是 "rule" 或 "mapping"。
    - source_set_method: 源节点的获取方法（如 "pid" 或 "set"）。
    - source_set_identifiers: 源节点的标识符列表。
    - target_set_method: 目标节点的获取方法（如 "pid" 或 "set"）。
    - target_set_identifiers: 目标节点的标识符列表。
    - mapping_file: 映射文件路径（仅当 task_type 为 "mapping" 时需要）。
    - id_replace_rule: 替换规则函数（仅当 task_type 为 "rule" 时需要）。
    - run_all_rules: 是否一次性执行所有规则（默认为 False）。
    - rules_to_run: 需要执行的规则名称列表（默认为 None）。
    """
    global global_reflection_plane

    # 如果对称面为空，则先计算对称面（只计算一次）
    if global_reflection_plane is None:
        set56_nodes = get_all_nodes(SET56_METHOD, SET56_IDENTIFIERS)
        normal, point_on_plane = calculate_dynamic_reflection_plane(set56_nodes)
        global_reflection_plane = (normal, point_on_plane)
        print("对称面已计算并存储：", global_reflection_plane)

    # 获取所有任务
    tasks = _get_all_tasks()

    # 如果 run_all_rules 为 True，则执行所有规则
    if run_all_rules:
        for task in tasks:
            print(f"\n执行任务：{task['name']}")
            _execute_reflect_task(task, global_reflection_plane)
        return

    # 如果 rules_to_run 不为空，则执行指定的规则
    if rules_to_run:
        for task in tasks:
            if task["name"] in rules_to_run:
                print(f"\n执行任务：{task['name']}")
                _execute_reflect_task(task, global_reflection_plane)
        return

    # 如果 run_all_rules 和 rules_to_run 均为 False/None，则执行单个任务
    if not all([task_type, source_set_method, source_set_identifiers, target_set_method, target_set_identifiers]):
        raise ValueError("如果 run_all_rules 和 rules_to_run 均为 False/None，则必须提供 task_type、source_set_method、source_set_identifiers、target_set_method、target_set_identifiers 参数。")

    # 执行单个任务
    task = {
        "task_type": task_type,
        "source_set_method": source_set_method,
        "source_set_identifiers": source_set_identifiers,
        "target_set_method": target_set_method,
        "target_set_identifiers": target_set_identifiers,
        'filter_by_prefix': filter_by_prefix,
        "mapping_file": mapping_file,
        "id_replace_rule": id_replace_rule
    }
    _execute_reflect_task(task, global_reflection_plane)

def _get_all_tasks():
    """
    内部函数：返回所有预定义的任务。
    """
    return [
        {
            "name": "手臂规则",  # 规则名称
            "task_type": "rule",
            "source_set_method": "pid",
            "source_set_identifiers": ["86200001", "86200301", "86200501", "86200801", "86201001"],
            "target_set_method": "pid",
            "target_set_identifiers": ["85200001", "85200301", "85200501", "85200801", "85201001"],
            "filter_by_prefix": True,
            "id_replace_rule": lambda x: int(str(x).replace("86", "85", 1))
        },
        {
            "name": "肩膀规则",  # 规则名称
            "task_type": "rule",
            "source_set_method": "pid",
            "source_set_identifiers": ["89200701"],
            "target_set_method": "pid",
            "target_set_identifiers": ["89700701"],
            "id_replace_rule": lambda x: x + 500000  # 替换规则：target 的 ID = source 的 ID + 500,000
        },
        {
            "name": "头部规则",  # 规则名称
            "task_type": "mapping",
            "source_set_method": "pid",
            "source_set_identifiers": ["88000230","88000229","88000231"],
            "target_set_method": "pid",
            "target_set_identifiers": ["88000222","88000221","88000223"],
            "mapping_file": "E://LZYZ//Scoliosis//RBF//Contest//final//node_mapping.csv"
        },
        {
            "name": "颈部规则",  # 规则名称
            "task_type": "rule",
            "source_set_method": "pid",
            "source_set_identifiers": ["87200101"],
            "target_set_method": "pid",
            "target_set_identifiers": ["87700101"],
            "filter_by_prefix": True,
            "id_replace_rule": lambda x: x + 500000
        },
        {
            "name": "胸部规则",  # 规则名称
            "task_type": "rule",
            "source_set_method": "pid",
            "source_set_identifiers": ["89200801"],
            "target_set_method": "pid",
            "target_set_identifiers": ["89700801"],
            "id_replace_rule": lambda x: x + 500000
        },
        {
            "name": "臀部规则",  # 规则名称
            "task_type": "rule",
            "source_set_method": "pid",
            "source_set_identifiers": ["83200101"],
            "target_set_method": "pid",
            "target_set_identifiers": ["83700101"],
            "id_replace_rule": lambda x: x + 500000
        },
        {
            "name": "腿部规则",  # 规则名称  
            "task_type": "rule",
            "source_set_method": "pid",
            "source_set_identifiers": ["82200001","82200401","82200601","82201101","82201301"],
            "target_set_method": "pid",
            "target_set_identifiers": ["81200001","81200401","81200601","81201101","81201301"],
            "filter_by_prefix": True,
            "id_replace_rule": lambda x: int(str(x).replace("82", "81", 1))
        },
    ]

def _execute_reflect_task(task, reflection_plane):
    """
    内部函数：执行单个反射任务。
    """
    filter_by_prefix = task.get("filter_by_prefix", False)
    source_set_nodes = get_all_nodes(task["source_set_method"], task["source_set_identifiers"], filter_by_prefix=filter_by_prefix)
    target_set_nodes = get_all_nodes(task["target_set_method"], task["target_set_identifiers"], filter_by_prefix=filter_by_prefix)

    # 使用全局存储的对称面
    normal, point_on_plane = reflection_plane
    # 获取对称面上的节点ID
    set56_nodes = get_all_nodes(SET56_METHOD, SET56_IDENTIFIERS)
    set56_node_ids = {node._id for node in set56_nodes}

    # 过滤掉对称面上的点（节点ID相同的点）
    filtered_source_nodes = [
        node for node in source_set_nodes
        if node._id not in set56_node_ids
    ]

    # 如果有点在对称面上，输出提示并跳过
    if len(filtered_source_nodes) < len(source_set_nodes):
        print(f"跳过 {len(source_set_nodes) - len(filtered_source_nodes)} 个对称面上的点。")
    # 提取源节点的坐标
    source_set_coords = np.array([node.position for node in filtered_source_nodes])

    # 反射变换
    reflected_coords = reflect_coordinates_with_near_plane_handling(source_set_coords, normal, point_on_plane)

    # 替换逻辑
    if task["task_type"] == "rule":
        set2_id_map = {node._id: node for node in target_set_nodes}
        for node, new_coord in zip(filtered_source_nodes, reflected_coords):
            set1_id = node._id
            set2_id = task["id_replace_rule"](set1_id)
            if set2_id in set2_id_map:
                set2_node = set2_id_map[set2_id]
                set2_node.position = tuple(new_coord)
            else:
                print(f"Set2 中未找到与 Set1 节点 ID {set1_id} 对应的节点 ID {set2_id}！")
    elif task["task_type"] == "mapping":
        try:
            df = pd.read_csv(task["mapping_file"])
            if "Left_Node_ID" not in df.columns or "Right_Node_ID" not in df.columns:
                raise ValueError("映射文件缺少必要的列：Left_Node_ID 或 Right_Node_ID")
            mapping = dict(zip(df["Left_Node_ID"], df["Right_Node_ID"]))
            print(f"从文件 {task['mapping_file']} 成功加载节点映射！")
        except Exception as e:
            print(f"加载映射文件失败: {e}")
            return

        set2_id_map = {node._id: node for node in target_set_nodes}
        for node, new_coord in zip(filtered_source_nodes, reflected_coords):
            set1_id = node._id
            set2_id = mapping.get(set1_id)
            if set2_id in set2_id_map:
                set2_node = set2_id_map[set2_id]
                set2_node.position = tuple(new_coord)
            else:
                print(f"Set2 中未找到与 Set1 节点 ID {set1_id} 对应的节点 ID {set2_id}！")
    else:
        print("未知任务类型！")

    print("对称替换完成！")

def is_point_on_plane(point, normal, point_on_plane, tolerance=1e-6):
    """
    判断一个点是否位于给定的平面上。
    使用点到平面的距离公式，若距离小于一个小的容忍值，则认为该点在平面上。
    """
    point_to_plane_distance = np.dot(np.array(point) - np.array(point_on_plane), normal)
    return abs(point_to_plane_distance) < tolerance

# 如果直接运行此脚本，则默认执行所有规则
if __name__ == "__main__":
    reflect(run_all_rules=True)