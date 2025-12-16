#-*- coding: utf-8 -*-
# 数据可视化
# 带文本散点图
import visdom
import numpy as np 

# 创建visdom可视化对象
vis=visdom.Visdom()  
vis.scatter(
    X=np.random.randn(6,2), 
    opts=dict(
        title='带文本散点图',
        textlabels=['Label%d' % i for i in range(6)],
        
    )
)  
