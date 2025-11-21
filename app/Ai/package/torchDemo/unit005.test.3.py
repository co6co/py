#-*- coding: utf-8 -*-
# 特征创建
# 使用 Pandas 特征创建  
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data={
    'A':[1,2,3],
    'B':[4,5,6],
    'C':[7,8,9]
}
df=pd.DataFrame(data)
#特征选择（一相关系数为例）
correlations=df.corr().abs()
print("相关系数矩阵:", correlations)
features=correlations[correlations['C']>0.5].index.tolist()
# 特征提取（以PCA为例）

pca=PCA(n_components=2)
df_pca=pca.fit_transform(df)
# 特征缩放
scaler=StandardScaler()
df_scaled=scaler.fit_transform(df)
print("原始数据:",df)
# 显示处理后的数据
print("特征选择结果:",features)
print("特征提取结果:",df_pca)
print("缩放后的数据:",df_scaled)

'''
原始数据:    A  B  C
0  1  4  7
1  2  5  8
2  3  6  9
特征选择结果: ['A', 'B', 'C']
特征提取结果: [[-1.73205081  0.        ]
 [ 0.          0.        ]
 [ 1.73205081  0.        ]]
缩放后的数据: [[-1.22474487 -1.22474487 -1.22474487]
 [ 0.          0.          0.        ]
 [ 1.22474487  1.22474487  1.22474487]]
'''