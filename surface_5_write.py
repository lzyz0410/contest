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

start_time = time.time()

# 根据配置获取目标节点
nodes_method = "csv"  # 获取节点的方式：从 CSV 获取 PID 列表
nodes_param = "E:\\LZYZ\\Scoliosis\\RBF\\Contest\\final\\shell_property.csv"
range = "A2:B35"  # 读取 PID 的范围
updated_node = get_all_nodes(nodes_method, nodes_param, range)
updated_node_data = np.array([[int(node._id)] + list(node.position) for node in updated_node])  # 将 nodeid 转换为整数

input_file_base = "E:\\LZYZ\\Scoliosis\\RBF\\Contest\\final\\THUMS_AM50_V402_Pedestrian_20150527_no_fracture2.k"
 # 加载原始文件内容并写入修改后的数据
with open(input_file_base, "r") as f:
    file_lines = f.readlines()

write_modified_coordinates(
    output_file="E:\\LZYZ\\Scoliosis\\RBF\\Contest\\final\\output.k",
    file_lines=file_lines,
    updated_node_data=updated_node_data
)
print("修改后的坐标已写入文件：output.k")
end_time = time.time()
print(f"文件名: {os.path.basename(__file__)}, 总运行时间: {end_time - start_time:.2f} s")