#-*- coding: utf-8 -*-
# 数据可视化
# 三维散点图
import visdom
import numpy as np 

# 创建visdom可视化对象
vis=visdom.Visdom()  
Y=np.random.randn(100) # 100个随机数组Y  
vis.scatter(
    X=np.random.randn(100,3), 
    Y=(Y%2).astype(int) +1,
    opts=dict(
        title='三维散点图',
        legend=['男性','女性'], 
    )
)  