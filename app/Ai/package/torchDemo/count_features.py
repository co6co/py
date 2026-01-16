import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 定义天干地支相关函数
tiangan = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']
dizhi = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']

def get_tiangan(year):
    return tiangan[(year - 4) % 10]

def get_dizhi(year):
    return dizhi[(year - 4) % 12]

def get_month_tiangan(year, month):
    month_tiangan_coeff = {'甲': [1, 4, 7, 10], '乙': [2, 5, 8, 11], '丙': [3, 6, 9, 12],
                          '丁': [1, 4, 7, 10], '戊': [2, 5, 8, 11], '己': [3, 6, 9, 12],
                          '庚': [1, 4, 7, 10], '辛': [2, 5, 8, 11], '壬': [3, 6, 9, 12],
                          '癸': [1, 4, 7, 10]}
    year_tg = get_tiangan(year)
    month_tg_order = ['丙', '丁', '戊', '己', '庚', '辛', '壬', '癸', '甲', '乙', '丙', '丁']
    start_idx = tiangan.index(year_tg)
    return month_tg_order[start_idx + month - 1 if start_idx + month - 1 < 10 else start_idx + month - 11]

def get_month_dizhi(month):
    return dizhi[(month + 1) % 12]

# 读取数据
region = pd.read_csv("./data/tmp/ssq.csv")
region_d = region[['QH','RQ']].copy()

# 数据预处理
region_d['RQ'] = pd.to_datetime(region_d['RQ'])

# 提取日期特征
region_d['Year'] = region_d['RQ'].dt.year
region_d['Month'] = region_d['RQ'].dt.month
region_d['Day'] = region_d['RQ'].dt.day
region_d['Hour'] = region_d['RQ'].dt.hour
region_d['Minute'] = region_d['RQ'].dt.minute
region_d['Weekday'] = region_d['RQ'].dt.weekday
region_d['Weekday_Name'] = region_d['RQ'].dt.day_name()
region_d['Is_Weekend'] = region_d['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
region_d['Quarter'] = region_d['RQ'].dt.quarter
region_d['DayOfYear'] = region_d['RQ'].dt.dayofyear
region_d['Timestamp'] = region_d['RQ'].astype('int64') // 10**9

# 添加特定日期特征
production_days = [0, 1, 2, 4]
region_d['Is_Production_Day'] = region_d['Weekday'].isin(production_days).astype(int)

# 添加时间间隔特征
region_d = region_d.sort_values('RQ')
region_d['Days_Since_Last'] = region_d['RQ'].diff().dt.days.fillna(0)
region_d['Hours_Since_Last'] = region_d['RQ'].diff().dt.total_seconds().fillna(0) / 3600

# 添加新的时间特征
region_d['Is_21h'] = (region_d['Hour'] == 21).astype(int)
region_d['Is_2115'] = ((region_d['Hour'] == 21) & (region_d['Minute'] >= 10) & (region_d['Minute'] <= 20)).astype(int)
region_d['Minutes_From_2115'] = abs((region_d['Hour'] - 21) * 60 + (region_d['Minute'] - 15))

# 添加更精细的星期特征
region_d['Is_Sunday'] = (region_d['Weekday'] == 6).astype(int)
region_d['Is_Monday'] = (region_d['Weekday'] == 0).astype(int)
region_d['Is_Tuesday'] = (region_d['Weekday'] == 1).astype(int)
region_d['Is_Thursday'] = (region_d['Weekday'] == 3).astype(int)

# 添加天干地支特征
region_d['Year_Tiangan'] = region_d['Year'].apply(get_tiangan)
region_d['Year_Dizhi'] = region_d['Year'].apply(get_dizhi)
region_d['Month_Tiangan'] = region_d.apply(lambda row: get_month_tiangan(row['Year'], row['Month']), axis=1)
region_d['Month_Dizhi'] = region_d['Month'].apply(get_month_dizhi)

# 处理分类特征
weekday_dummies = pd.get_dummies(region_d['Weekday_Name'], prefix='Weekday')
region_d = pd.concat([region_d, weekday_dummies], axis=1)
region_d = region_d.drop('Weekday_Name', axis=1)

# 处理天干地支独热编码
year_tg_dummies = pd.get_dummies(region_d['Year_Tiangan'], prefix='Year_Tiangan')
region_d = pd.concat([region_d, year_tg_dummies], axis=1)
region_d = region_d.drop('Year_Tiangan', axis=1)

year_dz_dummies = pd.get_dummies(region_d['Year_Dizhi'], prefix='Year_Dizhi')
region_d = pd.concat([region_d, year_dz_dummies], axis=1)
region_d = region_d.drop('Year_Dizhi', axis=1)

month_tg_dummies = pd.get_dummies(region_d['Month_Tiangan'], prefix='Month_Tiangan')
region_d = pd.concat([region_d, month_tg_dummies], axis=1)
region_d = region_d.drop('Month_Tiangan', axis=1)

month_dz_dummies = pd.get_dummies(region_d['Month_Dizhi'], prefix='Month_Dizhi')
region_d = pd.concat([region_d, month_dz_dummies], axis=1)
region_d = region_d.drop('Month_Dizhi', axis=1)

# 删除RQ列，只保留数值特征
region_d_numeric = region_d.drop('RQ', axis=1)

print(f'输入特征列数: {region_d_numeric.shape[1]}')
print(f'输入特征列名: {list(region_d_numeric.columns)}')