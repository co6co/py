#-*- coding: utf-8 -*-
# 1. 数据预处理
# 2. 常用工具 NumPy、Pandas、Scikit-learn, Matplotlib, Seaborn
import matplotlib # 用于绘制图形
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] # 用于显示中文字符
matplotlib.rcParams['axes.unicode_minus'] = False # 用于显示负号，避免出现乱码
x=[1,2,3,4,5]
y=[2,4,6,8,10]
s=input("0: 绘制散点图, 1: 绘制折线图, 2: 绘制柱状图, 3: 绘制直方图, 4: 绘制饼图, 5: 绘制箱线图, 6: 绘制3D图, 7: 绘制子图:")
if s=="0":
    plt.scatter(x,y) # 绘制散点图
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("散点图")
elif s=="1":
    plt.plot(x,y) # 绘制折线图
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("折线图")
elif s=="2":
    plt.bar(x,y) # 绘制柱状图
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("柱状图")
elif s=="3":
    plt.hist(x) # 绘制直方图
    plt.xlabel("x")
    plt.ylabel("Frequency")
    plt.title("直方图")
elif s=="4":
    plt.pie(y, labels=x) # 绘制饼图
    plt.title("饼图")        
    plt.legend()# 显示图   
elif s=="5":
    plt.boxplot(x) # 绘制箱线图
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("箱线图")
elif s=="6":
    fig=plt.figure()
    ax=fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,zs=0, zdir='z')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title("3D 散点图")
elif s=="7":
    plt.subplot(2,2,1)
    plt.scatter(x,y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("散点图")
    plt.subplot(2,2,2)
    plt.plot(x,y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("折线图")
    plt.subplot(2,2,3)
    plt.bar(x,y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("柱状图")
    plt.subplot(2,2,4)
    plt.hist(x)
    plt.xlabel("x")
    plt.ylabel("Frequency")
    plt.title("直方图") 

plt.show()