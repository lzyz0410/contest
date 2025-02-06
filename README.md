条件：ansa24.1.2+python3.11.9
所有的k文件都要先经过lsprepost导出，确保格式正确

icp_try.py进行尝试效果不好，还是rbf

本地运行记得改python外部库的位置



- 将base_control_set.k、target_modi.k和THUMS_AM50_V402_Pedestrian_20150527_no_fracture2.key一起导入ansa
  - 运行surface_1_joint_allrotate.py旋转手部的表面和内部(弃用运行surface_1_joint_rotate.py旋转左肘关节、左肩关节、并平滑达到近似目标姿态，根据左边手臂节点做对称变换)，手动导出THUMS_rotate.k作为bone的基准文件
  
  - 运行surface_2_process_rbf.py将基准体表映射到目标模型，再进行一次全身对称变换
  
  - 运行surface_3_process_rbf_uniform.py将基准体表二次分块映射到目标模型，除了手和手腕，变形太大，还没做
  
  - 运行surface_4_body_smooth.py，将表面进一步平滑对称，后续可以优化utils_smooth算法，平滑多了有向内收缩趋势
  
  - 直接保存成output.k
    - (弃用##运行surface_5_write.py，将表面节点坐标替换到原始thums文件中)，手动导出output.k
    
    

bone

- 同时打开base_control_set.k和THUMS_rotate.k，运行bone.py进行骨骼定位，和软组织映射，自动导出output_transformed.k，为最终的模型，运行154.1s