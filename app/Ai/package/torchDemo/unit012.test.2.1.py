#-*- coding: utf-8 -*-
# 数据可视化
# 绘制实线、虚线等
from matplotlib import markers
import visdom
import numpy as np 

# 创建visdom可视化对象
vis=visdom.Visdom()   
win=vis.line(
    #将三个在0和10 之间的等差数列组成一个3列矩阵
    X=np.column_stack((
      (np.arange(0,10)),
      (np.arange(0,10)),
      (np.arange(0,10)), 
    )),  
    # Y 轴数据，将Y的平方和Y+5 的平方根组合成一个2列矩阵
    Y=np.column_stack((
      np.linspace(5,10,10),
      np.linspace(5,10,10)+5,
      np.linspace(5,10,10)+10,
    )), 
    opts= {
      'dash':np.array(['solid','dash','dashdot']),
      'linecolor':np.array([[0,191,255],[0,255,0],[255,0,0]]),
    }
)
# 在之前的窗口上添加新的线
vis.line(
    X= (np.arange(0,10)),  
    Y= np.linspace(5,10,10)+15,
    name='Y+15', # 线段名称
    update='insert',
    win=win,
    opts= {
      'dash':np.array(['dot']),
      'linecolor':np.array([[0,100,255]]),
    }
)