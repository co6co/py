#-*- coding: utf-8 -*-
# 建立模型的一般步骤
# 示例
# 预测莫地区的最高温度
import torch
import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator # 用于设置坐标轴的刻度间隔
from sklearn import preprocessing #数据预处理

'''
year: 年	month: 月	day: 日	temp_2: 前天最高温度	temp_1: 昨天最高温度	average: 三年中这天平均温度	actual: 实际最高温度
'''
file    =".\\data\\005.weather.csv"
df=pd.read_csv(file)
labels=np.array(df['actual']) # 实际最高温度
df=df.drop('actual', axis=1) # 删除actual列
features_list=list(df.columns) # 列名列表
# 为提高模型准确率，对数据进行格式转换与标准化出来
features=np.array(df)
# 3. 通过特征标准化，不同特征的量级被消除，
#有助于提高机械选项算法的性能和准确行，
# 使模型在处理数据时不再受特征值大小的影响，能够公平地对待每个特征。
input_features=preprocessing.StandardScaler().fit_transform(features) # 标准化
# 4. 设置网络模型网络结构
input_size=input_features.shape[1] # 获取输入特征维度
hidden_size=128 # 隐藏层大小
output_size=1 # 输出层大小
batch_size=16 # 批量大小
# 使用 torch.nn.Sequential 按顺序构建神经网络模型
# 构建一个包含输入层、隐藏层和输出层的简单神经网络模型
# 后续可使用这个模型进行训练和预测操作
my_nn=torch.nn.Sequential(
    #线性层，将输入特征映射到隐层层
    torch.nn.Linear(input_size, hidden_size), 
    torch.nn.Sigmoid(),# 激活函数
    # 线性层，将隐层映射到输出层
    torch.nn.Linear(hidden_size, output_size)
)
# 5. 定义模型损失函数和优化器
# 训练过程中优化器将根据损失函数的反馈来调整神经网络参数，以最小化损失，这样通过不断迭代更新参数来改进模型性能
# 均方误差损失函数 
# reduction='mean' 表示对每个样本的损失进行平均，得到总体的平均损失
cost=torch.nn.MSELoss(reduction='mean')
# Adam优化器，传入神经网络参数学习率为0.001
optimizer=torch.optim.Adam(my_nn.parameters(), lr=0.001)

# 6. 训练神经网络模型
losses=[] # 存储损失值的列表
for i in range(500):
    # 用于存储每批数据的损失值
    batch_loss=[]
    # 遍历输入特征，按照批次大写进行分割
    for start in range(0, len(input_features), batch_size):
        # 计算批次的结束位置，如果超过特征长度则取特征长度
        end=start+batch_size if start+batch_size<len(input_features) else len(input_features)
        # 输入值转 tensor，设置类型flooat，需要计算梯度
        xx=torch.tensor(input_features[start:end], dtype=torch.float32, requires_grad=True)
        yy=torch.tensor(labels[start:end], dtype=torch.float32, requires_grad=True)
        prediction=my_nn(xx) # 使用构建的神经网络进行预测
        # 计算损失
        loss=cost(prediction, yy)
        # 清空梯度
        optimizer.zero_grad() 
        # 反向传播损失
        loss.backward(retain_graph=True)
        # 更新参数
        optimizer.step()
        batch_loss.append(loss.data.numpy()) # 记录损失值

        if i%100==0:
            losses.append(np.mean(batch_loss))
            print("Epoch: {}, Loss: {:.4f}".format(i, np.mean(batch_loss)))
    # 将输入特征转换为 torch.Tensor
    x=torch.tensor(input_features,dtype=torch.float32)
    predict=my_nn(x).data.numpy() #使用神经网络对X进行预测，并将结果转换为NumPy数组

    # 转换数据集中的日期格式
    # 将特征矩阵中的年份、月份和日期转换为字符串格式
    months=features[:,features_list.index('month')]
    days=features[:,features_list.index('day')]
    years=features[:,features_list.index('year')]
    dates=[str(int(year))+"-"+str(int(month))+"-"+str(int(day)) for year, month, day in  zip(years, months, days)]   
    dates=[datetime.datetime.strptime(date, "%Y-%m-%d") for date in dates]
    true_data=pd.DataFrame(data={'date':dates,'actual':labels})

    # 转换测试数据集中的日期格式
    test_dates=[str(int(year))+"-"+str(int(month))+"-"+str(int(day)) for year, month, day in  zip(years, months, days)]   
    test_dates=[datetime.datetime.strptime(date, "%Y-%m-%d") for date in test_dates]

    #创建包含保护测试日期和预测值的DF

    predictions_data=pd.DataFrame(data={'date':test_dates,'prediction':predict.reshape(-1)})

# 使用Matplotlib 绘制日最高温度的散点图
matplotlib.rc('font',family='SimHei')
plt.figure(figsize=(12,6),dpi=160)
# 真实值曲线 蓝色
plt.plot(true_data['date'], true_data['actual'], label='真实值', color='blue', alpha=0.5,marker='B+')
plt.plot(predictions_data['date'], predictions_data['prediction'], label='预测值', color='red', alpha=0.5,marker='o')
plt.xticks(rotation=30,size=15) # x 轴刻度旋转30度，大小15
plt.ylim(0,25) # y 轴范围 0-25
plt.yticks(size=15) # y 轴刻度大小15
x_major_locator=MultipleLocator(3) # x 轴主刻度间隔3
y_major_locator=MultipleLocator(5) # y 轴主刻度间隔5
ax=plt.gca() # 获取当前轴
ax.xaxis.set_major_locator(x_major_locator) # 设置x轴主刻度定位器
ax.yaxis.set_major_locator(y_major_locator) # 设置y轴主刻度定位器
plt.legend(fontsize=15) # 显示图例，字体大小15
plt.ylabel('日最高温度',size=15) # y 轴标签，大小15
plt.show() 
