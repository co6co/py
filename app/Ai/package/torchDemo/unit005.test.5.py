#-*- coding: utf-8 -*-
# 加载一个CSV文件，使用Pandas对数据进行清洗，包括缺失值、重复值和异常值
import pandas as pd
file    =".\\data\\005.weather.csv"
df=pd.read_csv(file)
print(df.head())
# 检查缺失值 
df.fillna(method='ffill', inplace=True)
df.drop_duplicates(inplace=True)
'''
   year  month  day  temp_2  temp_1  average  actual
0  2020      1  1.0   22.48   19.31    21.94   23.15
1  2020      1  2.0   19.31   23.15    19.30   21.52
2  2020      1  3.0   23.15   21.52    24.74   24.02
3  2020      1  4.0   21.52   24.02    18.59   23.48
4  2020      1  5.0   24.02   23.48    18.61   21.09
'''
# 检查异常值
m_cols=df.select_dtypes(include=['number']).columns

for col in m_cols:
    if col=='average':
        continue
        #df[col]=df[col].apply(lambda x:np.nan if x<0 else x) 
    else:
        df[col]=df[col].apply(lambda x:np.nan if x<0 else x) 
    
 
print(df.tail(30))