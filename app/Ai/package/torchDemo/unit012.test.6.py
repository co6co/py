#-*- coding: utf-8 -*-
# 数据可视化
# 热力图
# 条状图
# 箱型图
# 曲面图
# 等高线
from dis import stack_effect
import visdom
import numpy as np  
import math
vis=visdom.Visdom() 
vis.heatmap(
    X=np.outer(np.arange(1,6),np.arange(1,11)),
    opts=dict(
        title='热力图',
        columnnames=['a','b','c','d','e','f','g','h','i','j'],
        rownames=['y1','y2','y3','y4','y5'],
        colormap='Viridis',
    )
)

## 条状图
vis=visdom.Visdom() 
vis.bar(
    X=np.random.rand(4,3),
    opts=dict(
        title='条状图',
        stacked=True,
        legend=['低','中','高'],
        rownames=['2017','2018','2019','2020'], 
    )
)

# 箱型图
vis=visdom.Visdom() 
vis.boxplot(
    X=np.random.rand(100,2),
    opts=dict(
        title='箱型图',
        legend=['2019年','2020年'],
    )
)

# 曲面图
vis=visdom.Visdom() 
X=np.random.rand(100,2)
X[:,1]+=1

vis.surf(
    X=X,
    opts=dict(
        colormap='Viridis',
        title='曲面图',
    )
)

# 等高线
x=np.tile(np.arange(1,81),(80,1))
y=x.transpose()
X=np.exp((((x-40)**2)+((y-40)**2))/-(20.0**2) )
vis=visdom.Visdom() 
vis.contour(
    X=X, 
    opts=dict(
        colormap='Viridis',
        title='等高线',
    )
)
