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


set_nodes = get_nodes_from_set([17])
enforce_coordinate_uniformity(set_nodes, axis='o', left_nodes=None, smoothing_factor=0.1)
laplacian_smoothing(["87200101", "87700101"], iterations=1, alpha=0)
reflect(rules_to_run=["颈部规则","胸部规则"])
laplacian_smoothing(["89200801", "89700801"], iterations=2, alpha=0)

