#-*- coding: utf-8 -*-
# 数据标准化
# 使用 Scikit-learn进行数据预处理
# pip install scikit-learn
import numpy as np
from sklearn.preprocessing import StandardScaler

# 使用 fit_transform 进行标准化
data=np.array([[1,2],[3,4],[5,6]])
scaler=StandardScaler()
scaled_data=scaler.fit_transform(data) # 使用 fit_transform 方法进行标准化
print("原始数据:",data)
print("标准化后的数据:",scaled_data) 
