#-*- coding: utf-8 -*-
# 数据可视化
# 堆叠区域
import visdom
import numpy as np 
Y=np.linspace(0,4,200)
vis=visdom.Visdom()   
win=vis.line(
    X=np.column_stack((Y,Y)),  
    Y=np.column_stack((np.sqrt(Y),np.sqrt(Y)+2)),  
    opts=dict(
        fillarea=True,
        showlegend=False,
        width=380,
        height=330,
        ytype='log',
        title='堆叠区域图',
        marginleft=30,
        marginright=30,
        margintop=30,
        marginbottom=30, 
    )   
)
 