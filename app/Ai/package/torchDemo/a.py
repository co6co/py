import  torch
import torch.nn as nn
import torch.utils.data as Data #加载数据集、进行预处理
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules import transformer
from copy import deepcopy as copy #用于深拷贝
from mode import init_matplotlib_params

# 添加天干地支特征提取功能
# 定义天干地支列表
tiangan = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']
dizhi = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']

# 定义年份对应的天干地支映射
def get_tiangan(year):
    """根据年份获取天干"""
    return tiangan[(year - 4) % 10]

def get_dizhi(year):
    """根据年份获取地支"""
    return dizhi[(year - 4) % 12]

def get_month_tiangan(year, month):
    """根据年份和月份获取月天干"""
    # 月天干系数表
    month_tiangan_coeff = {'甲': [1, 4, 7, 10], '乙': [2, 5, 8, 11], '丙': [3, 6, 9, 12],
                          '丁': [1, 4, 7, 10], '戊': [2, 5, 8, 11], '己': [3, 6, 9, 12],
                          '庚': [1, 4, 7, 10], '辛': [2, 5, 8, 11], '壬': [3, 6, 9, 12],
                          '癸': [1, 4, 7, 10]}
    year_tg = get_tiangan(year)
    # 月天干顺序
    month_tg_order = ['丙', '丁', '戊', '己', '庚', '辛', '壬', '癸', '甲', '乙', '丙', '丁']
    start_idx = tiangan.index(year_tg)
    return month_tg_order[start_idx + month - 1 if start_idx + month - 1 < 10 else start_idx + month - 11]

def get_month_dizhi(month):
    """根据月份获取月地支"""
    return dizhi[(month + 1) % 12]

# 1. 读取数据
# 使用的模拟数据
region =pd.read_csv("./data/tmp/ssq.csv")

# 2. 数据准备
# 选取指定的两列作为基础特征，并创建副本避免SettingWithCopyWarning
region_d=region[['QH','RQ']].copy() 

# 3. 数据预处理
# 将RQ列（日期字符串）转换为datetime类型
region_d['RQ'] = pd.to_datetime(region_d['RQ'])

# 4. 从日期中提取必要特征
# 提取年份
region_d['Year'] = region_d['RQ'].dt.year
# 提取月份
region_d['Month'] = region_d['RQ'].dt.month
# 提取日期
region_d['Day'] = region_d['RQ'].dt.day
# 提取星期几（0=周一，6=周日）
region_d['Weekday'] = region_d['RQ'].dt.weekday

# 5. 固定时间为21:15:00（设备产生数据的固定时间）
region_d['Hour'] = 21  # 固定小时为21
region_d['Minute'] = 15  # 固定分钟为15
region_d['Is_2115'] = 1  # 标记为21:15产生的数据

# 7. 添加更多日期特征
# 月份的正弦余弦编码（捕捉周期性）
region_d['Month_sin'] = np.sin(2 * np.pi * region_d['Month'] / 12)
region_d['Month_cos'] = np.cos(2 * np.pi * region_d['Month'] / 12)

# 星期几的独热编码
weekday_dummies = pd.get_dummies(region_d['Weekday'], prefix='Weekday')
region_d = pd.concat([region_d, weekday_dummies], axis=1)

# 确保所有星期几列都存在
for i in range(7):
    col = f'Weekday_{i}'
    if col not in region_d.columns:
        region_d[col] = 0

# 7. 移除不需要的特征，只保留核心特征用于预测H1-H6
# 保留的特征：QH（流水号）、基本日期时间特征、21:15标记、月份周期性特征、星期几独热编码
keep_columns = ['QH', 'Year', 'Month', 'Day', 'Weekday', 'Hour', 'Minute', 'Is_2115', 
                'Month_sin', 'Month_cos',
                'Weekday_0', 'Weekday_1', 'Weekday_2', 'Weekday_3', 'Weekday_4', 'Weekday_5', 'Weekday_6']
region_d = region_d[keep_columns]

# 7. 选取需要预测的列作为目标
target_columns=['H1','H2','H3','H4','H5','H6','L']
target=region[target_columns]

# 查看L值的分布情况
print(f"L值的统计信息:")
print(target['L'].describe())
print(f"L值的唯一值: {target['L'].unique()}")
print(f"L值的分布: {target['L'].value_counts().sort_index()}") 

# 4. 数据归一化（可选，根据需要调整）
from sklearn.preprocessing import MinMaxScaler

# 只保留数值特征（RQ列已经被筛选掉了）
region_d_numeric = region_d.copy()

# 打印数据列数，用于调试
print(f"输入特征列数: {region_d_numeric.shape[1]}")
print(f"输入特征列名: {list(region_d_numeric.columns)}")

# 合并模型使用原始特征（56个）作为输入
input_data = region_d_numeric.values

# 分别对输入和目标进行归一化
scaler_input = MinMaxScaler()
scaler_target = MinMaxScaler()

# 转换为numpy数组
input_data = scaler_input.fit_transform(input_data)
target_data = scaler_target.fit_transform(target.values)

# 查看归一化前后的目标值统计信息
print(f"归一化前目标值的最小值: {target.min().values}")
print(f"归一化前目标值的最大值: {target.max().values}")
print(f"归一化后目标值的最小值: {target_data.min(axis=0)}")
print(f"归一化后目标值的最大值: {target_data.max(axis=0)}")

# 4. 划分训练集和测试集
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    input_data, target_data, test_size=0.2, random_state=42
)

# 5. 转换为torch张量
batch_size=30

# 创建训练和测试数据集
train_dataset=Data.TensorDataset(
    torch.tensor(train_input, dtype=torch.float32),
    torch.tensor(train_target, dtype=torch.float32)
)

test_dataset=Data.TensorDataset(
    torch.tensor(test_input, dtype=torch.float32),
    torch.tensor(test_target, dtype=torch.float32)
)

# 6. 创建数据加载器
train_loader=Data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
test_loader=Data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=0)


# 创建两个独立的模型：先预测H1-H6，再用H1-H6预测L值

# H1-H6预测模型（回归模型）
class HModel(nn.Module):
    """
    定义一个神经网络模型，用于预测H1-H6
    - 输入特征数量为17
    - 输出6个H值
    """
    def __init__(self):
        super(HModel,self).__init__()
        
        # H1-H6预测模型
        self.h_model = nn.Sequential(
            nn.Linear(17, 32),  # 输入特征17个，隐藏层32个神经元
            nn.ReLU(),  # ReLU激活函数
            nn.Dropout(0.3),  # Dropout防止过拟合
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 6)  # 输出6个H值
        )
    
    def forward(self, input: torch.FloatTensor):
        """
        前向传播函数，预测H1-H6
        
        @param input: 输入张量，形状为 (batch_size, 17)
        @return: H1-H6的预测结果，形状为 (batch_size, 6)
        """
        h_pred = self.h_model(input)
        return h_pred

# L值预测模型（分类模型）
class LModel(nn.Module):
    """
    定义一个神经网络模型，用于预测L值
    - 输入特征数量为17（仅使用原始特征，不依赖H值）
    - 输出16个类别的概率
    """
    def __init__(self):
        super(LModel,self).__init__()
        
        # L值预测模型 - 更深的独立架构
        self.l_model = nn.Sequential(
            nn.Linear(17, 128),  # 输入特征17个，隐藏层128个神经元
            nn.ReLU(),  # ReLU激活函数
            nn.BatchNorm1d(128),  # 添加BatchNorm
            nn.Dropout(0.5),  # Dropout防止过拟合
            nn.Linear(128, 64),  # 隐藏层64个神经元
            nn.ReLU(),
            nn.BatchNorm1d(64),  # 添加BatchNorm
            nn.Dropout(0.5),
            nn.Linear(64, 32),  # 隐藏层32个神经元
            nn.ReLU(),
            nn.BatchNorm1d(32),  # 添加BatchNorm
            nn.Dropout(0.5),
            nn.Linear(32, 16)  # 输出16个类别的概率
        )
    
    def forward(self, input: torch.FloatTensor, h_values: torch.FloatTensor = None):
        """
        前向传播函数，预测L值
        
        @param input: 原始输入张量，形状为 (batch_size, 17)
        @param h_values: 已弃用，保留参数以保持接口兼容
        @return: L值的预测结果，形状为 (batch_size, 16)（分类概率）
        """
        # 仅使用原始输入特征，不依赖H值预测
        l_pred = self.l_model(input)
        return l_pred

# 初始化模型
h_model = HModel()
l_model = LModel()

# 模型保存路径
h_model_path = './data/h_model.pth'
l_model_path = './data/l_model.pth'

# 定义损失函数
# H模型使用MSE损失函数
h_loss_func = nn.MSELoss()
# L模型使用交叉熵损失函数
l_loss_func = nn.CrossEntropyLoss()

# 定义优化器
h_optim = torch.optim.Adam(h_model.parameters(), lr=0.0001, weight_decay=1e-3)
l_optim = torch.optim.Adam(l_model.parameters(), lr=0.001, weight_decay=1e-4)  # 调整L模型的学习率和权重衰减

# 移除学习率调度器，使用固定低学习率

def tran_model(early_stop=5):
    """训练两个独立模型：先训练HModel，再训练LModel"""
    # 保存最佳模型
    best_h_model = None
    best_l_model = None
    best_total_loss = float('inf')
    
    # 第一阶段：只训练H模型
    print("\n=== 第一阶段：训练H模型 ===")
    h_epoch_cnt = 0
    h_best_loss = float('inf')
    
    for epoch in range(100):
        # 训练阶段
        h_model.train()
        total_h_train_loss = 0
        total_train_num = 0
        
        for batch_x, batch_y in train_loader:
            # 训练HModel预测H1-H6
            h_optim.zero_grad()
            h_pred = h_model(batch_x)
            h_loss = h_loss_func(h_pred, batch_y[:, :6])
            h_loss.backward()
            h_optim.step()
            
            total_h_train_loss += h_loss.item()
            total_train_num += 1
        
        # 计算平均训练损失
        avg_h_train_loss = total_h_train_loss / total_train_num if total_train_num > 0 else 0
        
        # 测试阶段
        h_model.eval()
        total_h_test_loss = 0
        total_test_num = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                # 使用HModel预测H1-H6
                h_pred = h_model(batch_x)
                h_loss = h_loss_func(h_pred, batch_y[:, :6])
                
                total_h_test_loss += h_loss.item()
                total_test_num += 1
        
        # 计算平均测试损失
        avg_h_test_loss = total_h_test_loss / total_test_num if total_test_num > 0 else 0
        
        if (epoch+1) % 10 == 0:
            print(f"H模型训练步骤{epoch+1}, H训练损失:{avg_h_train_loss:.6f}, H测试损失:{avg_h_test_loss:.6f}")
        
        # 保存最佳H模型
        if avg_h_test_loss < h_best_loss:
            h_best_loss = avg_h_test_loss
            best_h_model = copy(h_model)
            h_epoch_cnt = 0
        else:
            h_epoch_cnt += 1
        
        if h_epoch_cnt >= early_stop:
            print(f"H模型Early stopping at epoch {epoch+1}, best H test loss: {h_best_loss:.6f}")
            break
    
    # 加载最佳H模型
    if best_h_model is not None:
        h_model.load_state_dict(best_h_model.state_dict())
    
    # 第二阶段：只训练L模型，完全独立，不依赖H值
    print("\n=== 第二阶段：训练L模型 ===")
    l_epoch_cnt = 0
    l_best_loss = float('inf')
    
    for epoch in range(200):
        # 训练阶段
        l_model.train()
        total_l_train_loss = 0
        total_train_num = 0
        
        for batch_x, batch_y in train_loader:
            # 直接训练LModel预测L值，不依赖H值
            l_optim.zero_grad()
            l_pred = l_model(batch_x)  # 只使用原始输入
            # 将L值从0-1范围转换为1-16类别索引
            l_target = (batch_y[:, 6] * 15 + 1).long() - 1  # 0-15
            l_loss = l_loss_func(l_pred, l_target)
            l_loss.backward()
            l_optim.step()
            
            total_l_train_loss += l_loss.item()
            total_train_num += 1
        
        # 计算平均训练损失
        avg_l_train_loss = total_l_train_loss / total_train_num if total_train_num > 0 else 0
        
        # 测试阶段
        l_model.eval()
        total_h_test_loss = 0
        total_l_test_loss = 0
        total_test_num = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                # 使用HModel预测H1-H6
                h_pred = h_model(batch_x)
                h_loss = h_loss_func(h_pred, batch_y[:, :6])
                
                # 使用LModel预测L值，不依赖H值
                l_pred = l_model(batch_x)
                l_target = (batch_y[:, 6] * 15 + 1).long() - 1  # 0-15
                l_loss = l_loss_func(l_pred, l_target)
                
                total_h_test_loss += h_loss.item()
                total_l_test_loss += l_loss.item()
                total_test_num += 1
        
        # 计算平均测试损失
        avg_h_test_loss = total_h_test_loss / total_test_num if total_test_num > 0 else 0
        avg_l_test_loss = total_l_test_loss / total_test_num if total_test_num > 0 else 0
        avg_total_test_loss = avg_h_test_loss + avg_l_test_loss
        
        if (epoch+1) % 10 == 0:
            print(f"L模型训练步骤{epoch+1}, L训练损失:{avg_l_train_loss:.6f}, L测试损失:{avg_l_test_loss:.6f}")
        
        # 保存最佳L模型
        if avg_l_test_loss < l_best_loss:
            l_best_loss = avg_l_test_loss
            best_l_model = copy(l_model)
            best_total_loss = avg_total_test_loss
            l_epoch_cnt = 0
        else:
            l_epoch_cnt += 1
        
        if l_epoch_cnt >= early_stop:
            print(f"L模型Early stopping at epoch {epoch+1}, best L test loss: {l_best_loss:.6f}")
            break
    
    # 保存最终最佳模型
    if best_h_model is not None and best_l_model is not None:
        torch.save(best_h_model.state_dict(), h_model_path)
        torch.save(best_l_model.state_dict(), l_model_path)
        print(f"\n模型保存成功，最佳总测试损失: {best_total_loss:.6f}")
        print(f"其中：H测试损失: {avg_h_test_loss:.6f}, L测试损失: {l_best_loss:.6f}")

def test_model():
    """测试模型的函数
    @return: 测试值、真实标签和测试损失
    """
    total_h_test_loss = 0  # H模型测试总损失
    total_l_test_loss = 0  # L模型测试总损失
    total_test_num = 0  # 测试样本总数
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            # 1. 使用HModel预测H1-H6
            h_pred = h_model(batch_x)
            h_loss = h_loss_func(h_pred, batch_y[:, :6])
            
            # 2. 使用LModel预测L值
            l_pred = l_model(batch_x, h_pred)
            l_target = (batch_y[:, 6] * 15 + 1).long() - 1  # 0-15
            l_loss = l_loss_func(l_pred, l_target)
            
            # 将L值的分类预测转换为具体数值（1-16）
            l_pred_class = torch.argmax(l_pred, dim=1) + 1  # 转换为1-16
            l_pred_class = l_pred_class.float().unsqueeze(1)  # 转换为float并添加维度
            
            # 合并预测结果
            pred = torch.cat([h_pred, l_pred_class], dim=1)
            
            total_h_test_loss += h_loss.item()
            total_l_test_loss += l_loss.item()
            total_test_num += 1
            
            all_preds.append(pred.numpy())
            all_labels.append(batch_y.numpy())
    
    # 计算平均损失
    avg_h_test_loss = total_h_test_loss / total_test_num if total_test_num > 0 else 0
    avg_l_test_loss = total_l_test_loss / total_test_num if total_test_num > 0 else 0
    total_test_loss = avg_h_test_loss + avg_l_test_loss  # 总损失为两者之和
    
    return all_preds, all_labels, total_test_loss

def plot_test_result(preds,labels,test_loss):
    """绘制测试结果的函数
    @param preds: 测试值列表
    @param labels: 真实标签列表
    @param test_loss: 测试损失
    """ 
    # 将批次结果合并
    import numpy as np
    all_preds = np.concatenate(preds, axis=0)
    all_labels = np.concatenate(labels, axis=0)
    
    # 对预测结果进行后处理，确保符合要求
    processed_preds = np.copy(all_preds)
    
    for i in range(processed_preds.shape[0]):
        # 处理H1-H6（1-33，无重复）
        used_values = set()
        for j in range(6):
            val = round(processed_preds[i, j])
            val = max(1, min(33, val))
            
            # 去重处理
            original_val = val
            max_iter = 33
            iter_count = 0
            
            while val in used_values and iter_count < max_iter:
                if val < 17:
                    val += 1
                else:
                    val -= 1
                val = max(1, min(33, val))
                iter_count += 1
            
            used_values.add(val)
            processed_preds[i, j] = val
        
        # 处理L值（1-16）
        l_val = round(processed_preds[i, 6])
        l_val = max(1, min(16, l_val))
        processed_preds[i, 6] = l_val
    
    # 绘制预测值与真实值的对比图
    plt.figure(figsize=(12,7),dpi=100) # 创建图形窗口
    plt.suptitle(f"预测结果对比 (测试损失: {test_loss:.6f})") # 设置图形标题
    init_matplotlib_params()
    # 绘制每个目标列的对比
    for i in range(processed_preds.shape[1]):
        plt.subplot(2, 4, i+1)
        # 生成横坐标（样本索引）
        x = range(len(all_labels[:, i]))
        plt.plot(x, all_labels[:, i], label=f"真实值", marker="X", markersize=4, linestyle="-")
        plt.plot(x, processed_preds[:, i], label=f"预测值", marker="o", markersize=4, linestyle="--")
        plt.title(f"{target_columns[i]}")
        plt.xlabel("样本索引")  # 添加横坐标标签
        plt.ylabel("数值")  # 添加纵坐标标签
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)  # 添加网格线，提高可读性
    
    plt.tight_layout()
    plt.show() # 显示图形
    

def predict_from_user_input():
    """从用户输入获取QH和RQ，返回预测结果[H1,H2,H3,H4,H5,H6,L]"""
    print("\n=== 用户输入预测 ===")
    try:
        # 获取用户输入
        qh = input("请输入QH值: ")
        rq = input("请输入RQ值 (格式: YYYY/MM/DD): ")
        
        # 验证并处理输入
        qh = int(qh)
        # 将字符串转换为datetime对象
        rq_dt = pd.to_datetime(rq)
        
        # 提取日期相关特征
        year = rq_dt.year
        month = rq_dt.month
        day = rq_dt.day
        weekday = rq_dt.weekday()  # 注意：weekday是一个方法，需要调用它来获取值
        
        # 固定时间为21:15:00
        hour = 21
        minute = 15
        is_2115 = 1
        
        # 构建基础特征向量
        base_features = np.array([[
            qh, year, month, day, weekday, hour, minute, is_2115
        ]])
        
        # 添加月份的正弦余弦编码
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        # 添加星期几的独热编码
        weekday_onehot = np.zeros(7)
        weekday_onehot[weekday] = 1
        
        # 构建完整的输入特征向量
        user_input = np.concatenate([
            base_features,
            np.array([[month_sin, month_cos]]),
            weekday_onehot.reshape(1, -1)
        ], axis=1)
        
        # 归一化输入
        normalized_input = scaler_input.transform(user_input)
        input_tensor = torch.tensor(normalized_input, dtype=torch.float32)
        
        # 使用模型进行预测
        h_model.eval()
        l_model.eval()
        with torch.no_grad():
            # 1. 预测H1-H6
            h_pred = h_model(input_tensor)
            # 2. 使用H1-H6预测L值
            l_pred = l_model(input_tensor, h_pred)
            # 将L值的分类预测转换为具体数值（1-16）
            l_pred_class = torch.argmax(l_pred, dim=1) + 1  # 转换为1-16
        
        # 合并预测结果
        normalized_pred = np.concatenate([h_pred.numpy(), l_pred_class.numpy().reshape(-1, 1)], axis=1)
        
        # 反归一化H1-H6
        pred = scaler_target.inverse_transform(normalized_pred)[0]
        
        # 处理预测结果：转换为1-33之间的整数，且H1-H6无重复
        used_values = set()
        processed_pred = []
        
        # 对H1-H6进行处理（1-33，无重复）
        for i in range(6):
            # 四舍五入到最近整数，然后限制在1-33之间
            val = round(pred[i] * 0.95)  # 乘以0.95的缩放因子，解决预测值偏高问题
            val = max(1, min(33, val))
            
            # 去重处理
            original_val = val
            max_iter = 33  # 避免无限循环
            iter_count = 0
            
            while val in used_values and iter_count < max_iter:
                # 如果值已存在，调整到最近的可用数值
                if val < 17:  # 如果值较小，增加1
                    val += 1
                else:  # 如果值较大，减少1
                    val -= 1
                # 确保值仍在1-33范围内
                val = max(1, min(33, val))
                iter_count += 1
            
            used_values.add(val)
            processed_pred.append(val)
        
        # 处理L值（1-16）
        l_val = int(pred[6])
        l_val = max(1, min(16, l_val))
        processed_pred.append(l_val)
        
        return processed_pred
    except ValueError as e:
        print(f"输入错误: {e}")
        return None
    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        return None


def test_with_csv_file():
    """使用ssq_test.csv文件进行测试"""
    print("\n=== 使用测试文件进行预测 ===")
    try:
        # 读取测试文件
        test_file_path = "./data/tmp/ssq_test.csv"
        test_data = pd.read_csv(test_file_path)
        print(f"读取测试数据成功，共 {test_data.shape[0]} 条记录")
        
        # 复制测试数据用于输出
        output_data = test_data.copy()
        
        # 数据预处理
        # 选取基础特征
        test_d = test_data[['QH', 'RQ']].copy()
        # 转换日期格式
        test_d['RQ'] = pd.to_datetime(test_d['RQ'])
        
        # 提取日期相关特征
        test_d['Year'] = test_d['RQ'].dt.year
        test_d['Month'] = test_d['RQ'].dt.month
        test_d['Day'] = test_d['RQ'].dt.day
        test_d['Weekday'] = test_d['RQ'].dt.weekday
        
        # 固定时间为21:15:00
        test_d['Hour'] = 21
        test_d['Minute'] = 15
        test_d['Is_2115'] = 1
        
        # 添加更多日期特征
        # 月份的正弦余弦编码
        test_d['Month_sin'] = np.sin(2 * np.pi * test_d['Month'] / 12)
        test_d['Month_cos'] = np.cos(2 * np.pi * test_d['Month'] / 12)
        
        # 星期几的独热编码
        weekday_dummies = pd.get_dummies(test_d['Weekday'], prefix='Weekday')
        test_d = pd.concat([test_d, weekday_dummies], axis=1)
        
        # 确保所有星期几列都存在
        for i in range(7):
            col = f'Weekday_{i}'
            if col not in test_d.columns:
                test_d[col] = 0
        
        # 只保留核心特征
        test_d = test_d[['QH', 'Year', 'Month', 'Day', 'Weekday', 'Hour', 'Minute', 'Is_2115', 
                        'Month_sin', 'Month_cos',
                        'Weekday_0', 'Weekday_1', 'Weekday_2', 'Weekday_3', 'Weekday_4', 'Weekday_5', 'Weekday_6']]
        
        # 转换为numpy数组
        test_input = test_d.values
        
        # 归一化输入
        normalized_test_input = scaler_input.transform(test_input)
        test_tensor = torch.tensor(normalized_test_input, dtype=torch.float32)
        
        # 使用模型进行预测
        h_model.eval()
        l_model.eval()
        with torch.no_grad():
            # 1. 预测H1-H6
            h_pred = h_model(test_tensor)
            # 2. 使用H1-H6预测L值
            l_pred = l_model(test_tensor, h_pred)
            # 将L值的分类预测转换为具体数值（1-16）
            l_pred_class = torch.argmax(l_pred, dim=1) + 1  # 转换为1-16
        
        # 合并预测结果
        h_pred_np = h_pred.numpy()
        l_pred_np = l_pred_class.numpy().reshape(-1, 1)
        normalized_pred = np.concatenate([h_pred_np, np.zeros_like(l_pred_np)], axis=1)  # 占位符L值
        
        # 反归一化H1-H6
        predictions = scaler_target.inverse_transform(normalized_pred)
        
        # 处理预测结果
        processed_predictions = []
        
        for i in range(predictions.shape[0]):
            pred = predictions[i]
            used_values = set()
            processed_pred = []
            
            # 处理H1-H6（1-33，无重复）
            for j in range(6):
                # 对预测值进行校准，使其更接近真实值范围
                val = round(pred[j] * 0.95)  # 乘以0.95的缩放因子，解决预测值偏高问题
                
                # 确保值在1-33范围内
                val = max(1, min(33, val))
                
                max_iter = 33
                iter_count = 0
                while val in used_values and iter_count < max_iter:
                    if val < 17:
                        val += 1
                    else:
                        val -= 1
                    val = max(1, min(33, val))
                    iter_count += 1
                
                used_values.add(val)
                processed_pred.append(val)
            
            # 处理L值（1-16）
            l_val = int(l_pred_np[i][0])
            l_val = max(1, min(16, l_val))
            processed_pred.append(l_val)
            
            processed_predictions.append(processed_pred)
        
        # 输出预测结果
        print("\n=== 测试文件预测结果 ===")
        print("QH, RQ, 真实值(H1-H6,L), 预测值(H1-H6,L)")
        for i in range(len(test_data)):
            qh = test_data.iloc[i]['QH']
            rq = test_data.iloc[i]['RQ']
            
            # 获取真实值
            real_values = test_data.iloc[i][['H1','H2','H3','H4','H5','H6','L']].tolist()
            real_str = ",".join([str(int(x)) for x in real_values])
            
            # 获取预测值
            pred_values = processed_predictions[i]
            pred_str = ",".join([str(x) for x in pred_values])
            
            print(f"{qh}, {rq}, {real_str}, {pred_str}")
        
        # 保存预测结果到文件
        output_file = "./data/tmp/ssq_test_predictions.csv"
        output_data['Pred_H1'] = [p[0] for p in processed_predictions]
        output_data['Pred_H2'] = [p[1] for p in processed_predictions]
        output_data['Pred_H3'] = [p[2] for p in processed_predictions]
        output_data['Pred_H4'] = [p[3] for p in processed_predictions]
        output_data['Pred_H5'] = [p[4] for p in processed_predictions]
        output_data['Pred_H6'] = [p[5] for p in processed_predictions]
        output_data['Pred_L'] = [p[6] for p in processed_predictions]
        output_data.to_csv(output_file, index=False)
        print(f"\n预测结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"测试文件处理失败: {e}")


if __name__ == "__main__":
    import argparse
    
    # 添加命令行参数
    parser = argparse.ArgumentParser(description='使用神经网络从QH和RQ预测H1-H6和L')
    parser.add_argument('--load-model', action='store_true', default=False,
                       help='加载已训练好的模型，而不是重新训练')
    parser.add_argument('--test-file', action='store_true', default=False,
                       help='使用ssq_test.csv文件进行批量测试')
    args = parser.parse_args()
    
    # 超参数
    early_stop=5
    
    if args.load_model:
        print("正在加载训练好的模型...")
        try:
            # 加载H模型
            h_model.load_state_dict(torch.load(h_model_path))
            # 加载L模型
            l_model.load_state_dict(torch.load(l_model_path))
            print("模型加载成功！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("将重新训练模型...")
            # 训练模型
            tran_model(early_stop)
    else:
        print("开始训练模型...")
        # 训练模型
        tran_model(early_stop)
    
    print("\n开始测试模型...")
    # 使用测试数据加载器进行测试
    test_pred,test_y,test_loss=test_model()
    
    print(f"\n测试完成，测试损失: {test_loss:.6f}")
    
    # 绘制测试结果
    plot_test_result(test_pred,test_y,test_loss)
    
    # 如果指定了--test-file参数，使用测试文件进行批量预测
    if args.test_file:
        test_with_csv_file()
    else:
        # 用户输入预测
        while True:
            prediction = predict_from_user_input()
            if prediction:
                print(f"\n预测结果: H1={prediction[0]}, H2={prediction[1]}, H3={prediction[2]}, H4={prediction[3]}, H5={prediction[4]}, H6={prediction[5]}, L={prediction[6]}")
            
            # 询问是否继续
            continue_input = input("\n是否继续预测？(y/n): ")
            if continue_input.lower() != 'y':
                break

    print("\n代码执行完成！")
