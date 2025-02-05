import os
import sys
third_packages = r"G:\\anaconda3\\envs\\ansa_meta_env\\Lib\\site-packages"
sys.path.append(third_packages)
import ansa
from ansa import *
import numpy as np
import time
from utils_node import *
from utils_smooth import *
from utils_reflect import *
from utils_adjust import *

start_time = time.time()
set_nodes = get_nodes_from_set([17])
enforce_coordinate_uniformity(get_nodes_from_set([17]), axis='o', left_nodes=None, smoothing_factor=0.1)            
reflect(rules_to_run=["肩胸规则"])

laplacian_smoothing(["87200101", "87700101","89200801", "89700801"], iterations=1, alpha=0)
laplacian_smoothing(["89200801", "89700801"], iterations=5, alpha=0)
laplacian_smoothing(["89200701","86200001"], iterations=2, alpha=0)
enforce_coordinate_uniformity(get_nodes_from_set([22]), axis='o', left_nodes=None, smoothing_factor=0.1)     
laplacian_smoothing(["89200701"], iterations=2, alpha=0)      
laplacian_smoothing(["86200001"], iterations=2, alpha=0)     
laplacian_smoothing(["86200001","86200301", "86200501","86200801"], iterations=2, alpha=0)
laplacian_smoothing(["86200801"], iterations=5, alpha=0)

reflect(rules_to_run=["颈部规则","肩胸规则","手部规则"])

laplacian_smoothing(["83200101"], iterations=5, alpha=0)
laplacian_smoothing(["82200001","82200401"], iterations=5, alpha=0)
enforce_coordinate_uniformity(get_nodes_from_set([12]), axis='z', left_nodes=None, smoothing_factor=0.1)            
laplacian_smoothing(["82200001"], iterations=2, alpha=0)
laplacian_smoothing(["82200401","82200601"], iterations=1, alpha=0)  #x
enforce_coordinate_uniformity(get_nodes_from_set([10]), axis='z', left_nodes=None, smoothing_factor=0.1)            
laplacian_smoothing(["82200601","82201101"], iterations=10, alpha=0)
enforce_coordinate_uniformity(get_nodes_from_set([8]), axis='z', left_nodes=None, smoothing_factor=0.1)            

reflect(rules_to_run=True)
end_time = time.time()
print(f"文件名: {os.path.basename(__file__)}, 总运行时间: {end_time - start_time:.2f} s")