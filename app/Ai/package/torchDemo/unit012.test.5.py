#-*- coding: utf-8 -*-
# 数据可视化
# 正弦函数和余弦函数
import visdom
import numpy as np  
import math
vis=visdom.Visdom()
Y=np.linspace(0,2*math.pi,70)
#将正弦函数和余弦函数的对应值堆叠在一起
X=np.column_stack((np.sin(Y),np.cos(Y)))

vis.stem(
    X=X,
    Y=Y,
    opts=dict(
        title='正弦函数和余弦函数',
        legend=['正弦函数','余弦函数'], 
    )
)
