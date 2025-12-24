#-*- coding: utf-8 -*-
# 四维空间散列图
import numpy as np
import visdom
import torch,sys

# 启动 visdom 服务器: python -m visdom.server
viz = visdom.Visdom()

# 生成四维数据 (x, y, z, w)
n_points = 200
data_4d = np.random.randn(n_points, 4)  # 四个维度的随机数据 
def standardize_then_scale(data,max=255):
    """先标准化为N(0,1)，再映射到0-255"""
    # 注意：randn已经是标准正态分布，这里演示一般流程
    mean_val = data.mean()
    std_val = data.std()
    
    # 标准化到 [0, 1] 然后扩展到更大范围
    standardized = (data - mean_val) / std_val  # 这会得到负值和正值
    
    # 将标准化后的值映射到 [0, 255]
    # 选择一个合适的范围，比如 ±3σ 覆盖99.7%的数据
    min_std = -3
    max_std = 3
    mapped = (standardized - min_std) / (max_std - min_std) * max
    mapped = np.clip(mapped, 0, max)  # 确保在范围内
    
    return mapped
#data_4d=standardize_then_scale(data_4d).astype(int)  
# 提取前三维坐标
x = data_4d[:, 0]
y = data_4d[:, 1] 
z = data_4d[:, 2]
w = data_4d[:, 3]  # 第四维用作颜色/大小  
c2=standardize_then_scale(np.random.rand(*w.shape) ).astype(int)  
points_line = np.column_stack([standardize_then_scale(w).astype(int) ,c2,np.full_like(w, 5)  ])
# 方法1: 用颜色表示第四维
scatter_color = viz.scatter(
    X=torch.FloatTensor(np.column_stack([x, y, z])),
    Y= torch.zeros(n_points, dtype=torch.int)+1, 
    opts=dict(
        markersize=5,
        #markercolor=np.clip(w, 0, 1),  # 将第四维映射到颜色
        markercolor=points_line,  # 将第四维映射到颜色
        markerborderwidth=0,
        legend=['Fourth dimension as color'],
        title='4D Scatter: Color represents 4th dimension',
        xlabel='X',
        ylabel='Y', 
        zlabel='Z'
    )
) 
# 方法2: 用点的大小表示第四维
scatter_size = viz.scatter(
    X=torch.FloatTensor(np.column_stack([x, y, z])),
    Y=torch.zeros(n_points, dtype=torch.int)+1,
    opts=dict(
        markersize=np.abs(w) * 10 + 2 ,  # 第四维控制点的大小
        #markercolor='blue',
        markerborderwidth=0,
        legend=['Fourth dimension as size'],
        title='4D Scatter: Size represents 4th dimension',
        xlabel='X',
        ylabel='Y',
        zlabel='Z'
    )
)