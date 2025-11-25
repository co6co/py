#-*- coding: utf-8 -*-
# 归一化
# 使用 Scikit-learn进行数据预处理
# pip install scikit-learn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
data=np.array([[10,2],[5,6],[3,8]])
scaler=MinMaxScaler()
normalized_data=scaler.fit_transform(data)
print("原始数据:",data)
print("归一化后的数据:",normalized_data)

