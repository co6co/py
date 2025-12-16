#-*- coding: utf-8 -*-
# 数据可视化
# 绘制直线
from matplotlib import markers
import visdom
import numpy as np 

# 创建visdom可视化对象
vis=visdom.Visdom()  
Y=np.linspace(-5,5,100)
vis.line(
    X=np.column_stack((Y,Y)), #自生组成2列矩阵
    # Y 轴数据，将Y的平方和Y+5 的平方根组合成一个2列矩阵
    Y=np.column_stack((Y*Y,np.sqrt(Y+5))),
    opts=dict(
      markers=False,
    )
)