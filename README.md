所有的k文件都要先经过lsprepost导出一般，确保格式正确

icp_try.py进行尝试效果不好，还是rbf



- 将base_control_set.k、target_modi.k和THUMS_AM50_V402_Pedestrian_20150527_no_fracture2.k一起导入ansa
  - 运行joint_rotate.py旋转左肘关节、左肩关节、并平滑达到近似目标姿态，根据左边手臂节点做对称变换
  - 运行process_rbf.py将基准体表映射到目标模型，再进行一次全身对称变换
  - 运行process_rbf_uniform.py将基准体表二次分块映射到目标模型，除了手和手腕，变形太大，还没做