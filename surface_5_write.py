import os
import sys
third_packages = r"G:\\anaconda3\\envs\\ansa_meta_env\\Lib\\site-packages"
sys.path.append(third_packages)
import ansa
from ansa import *
import numpy as np
import time
from utils_node import *
from utils_data import *
from utils_env import *

start_time = time.time()

# **动态查找 `shell_property.csv`**
shell_property_path = find_file_in_parents("shell_property.csv")

# **动态查找 `THUMS_AM50_V402_Pedestrian_20150527_no_fracture2.k`**
input_file_base = find_file_in_parents("THUMS_AM50_V402_Pedestrian_20150527_no_fracture2.k")

# **动态设置 `output.k` 存放位置**
output_file = Path(__file__).resolve().parent / "output.k"

# **根据配置获取目标节点**
nodes_method = "csv"  # 获取节点的方式：从 CSV 获取 PID 列表
range = "A2:B35"  # 读取 PID 的范围
updated_node = get_all_nodes(nodes_method, str(shell_property_path), range)
updated_node_data = np.array([[int(node._id)] + list(node.position) for node in updated_node])  # 将 nodeid 转换为整数

# **加载原始文件内容**
with open(input_file_base, "r") as f:
    file_lines = f.readlines()

# **写入修改后的数据**
write_modified_coordinates(
    output_file=str(output_file),
    file_lines=file_lines,
    updated_node_data=updated_node_data
)

print(f"修改后的坐标已写入文件: {output_file}")
end_time = time.time()
print(f"文件名: {os.path.basename(__file__)}, 总运行时间: {end_time - start_time:.2f} s")