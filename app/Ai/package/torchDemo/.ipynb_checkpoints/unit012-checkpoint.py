#-*- coding: utf-8 -*-
# 数据可视化
# 普通散点图
import visdom
import numpy as np

# 创建visdom可视化对象
vis=visdom.Visdom()
Y=np.random.randn(100) # 100个随机数组Y
# 使用visdom的scatter函数绘制散点图，传入x轴和Y轴的数据
# 确保Y的长度与X一致，将Y转换为0或1的标签
labels = (Y > 0).astype(int) + 1  # 将Y分为两类：1和2
win=vis.scatter(
    X=np.random.randn(100,2),
    Y=labels,
    opts=dict(
        title='普通散点图',
        legend=['Didnt','Update'],
        xtickmin=-50,
        xtickmax=50,
        xtickstep=0.5,
        ytickmin=-50,
        ytickmax=50,
        ytickstep=0.5,
        markersymbol='cross-thin-open',
    )
)
# 使用update_windows_opts 函数更新之前没绘制的散点图配置选项
vis.update_window_opts(
    win=win, 
    opts=dict(
        title='普通散点图', 
        legend=['2019年','2020年'],
        xtickmin=0,
        xtickmax=1,
        xtickstep=0.5,
        ytickmin=0,
        ytickmax=1,
        ytickstep=0.5,
        markersymbol='cross-thin-open',
    )
)
