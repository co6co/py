import numpy as np
import pandas as pd
# pip install scipy scikit-learn matplotlib
from scipy import stats, fft
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


class MouseMovementAnalyzer:
    def __init__(self, data):
        """
        data: DataFrame 包含 columns ['timestamp', 'x', 'y']
        """
        self.data = data.sort_values('timestamp').reset_index(drop=True)
        self.calculate_basic_features()

    def calculate_basic_features(self):
        # 时间差 (毫秒)
        self.data['dt'] = self.data['timestamp'].diff().fillna(0)

        # 位移计算
        dx = self.data['x'].diff().fillna(0)
        dy = self.data['y'].diff().fillna(0)
        self.data['distance'] = np.sqrt(dx**2 + dy**2)

        # 瞬时速度 (像素/毫秒)
        self.data['velocity'] = self.data['distance'] / (self.data['dt'] + 1e-6)  # 避免除零

        # 加速度计算
        dv = self.data['velocity'].diff().fillna(0)
        self.data['acceleration'] = dv / (self.data['dt'] + 1e-6)

    def speed_analysis(self):
        """速度分布特征"""
        v = self.data['velocity']
        return {
            'velocity_std': v.std(),
            'velocity_skew': v.skew(),
            'acceleration_std': self.data['acceleration'].std(),
            'zero_acceleration_ratio': (self.data['acceleration'].abs() < 1e-3).mean()
        }

    def trajectory_analysis(self, window=5):
        """轨迹曲率分析"""
        x, y = self.data['x'].values, self.data['y'].values
        curvature = []

        for i in range(1, len(x)-1):
            dx1, dy1 = x[i] - x[i-1], y[i] - y[i-1]
            dx2, dy2 = x[i+1] - x[i], y[i+1] - y[i]
            angle = np.arctan2(dy2, dx2) - np.arctan2(dy1, dx1)
            curvature.append(abs(angle))

            fft_coeff = np.abs(fft.fft(curvature)[:len(curvature)//2])
            return {
                'curvature_std': np.std(curvature),
                'fft_peak_freq': np.argmax(fft_coeff) / len(curvature),
                'trajectory_smoothness': 1 / (np.mean(curvature) + 1e-6)
            }

    def temporal_analysis(self):
        """时间间隔分析"""
        dt = self.data['dt']
        click_indices = self.data[self.data['distance'] == 0].index  # 假设静止为点击

        pre_click_speed = []
        for idx in click_indices:
            if idx > 0:
                pre_click_speed.append(self.data.loc[idx-1, 'velocity'])

        return {
            'dt_entropy': stats.entropy(np.histogram(dt, bins=20)[0]),
            'click_pre_speed_mean': np.mean(pre_click_speed) if pre_click_speed else 0
        }

    def spatial_analysis(self):
        """空间分布特征"""
        x, y = self.data['x'], self.data['y']
        hist, _, _ = np.histogram2d(x, y, bins=(10, 10))
        return {
            'spatial_entropy': stats.entropy(hist.flatten()),
            'cluster_score': np.max(hist) / np.sum(hist)
        }

    def extract_all_features(self):
        features = {}
        features.update(self.speed_analysis())
        features.update(self.trajectory_analysis())
        features.update(self.temporal_analysis())
        features.update(self.spatial_analysis())
        return pd.DataFrame([features])

    def visualize(self):
        """可视化轨迹和速度分布"""
        plt.figure(figsize=(12, 4))

        plt.subplot(131)
        plt.plot(self.data['x'], self.data['y'], '-o', markersize=2)
        plt.title('Movement Trajectory')

        plt.subplot(132)
        plt.hist(self.data['velocity'], bins=50)
        plt.title('Velocity Distribution')

        plt.subplot(133)
        plt.scatter(self.data['x'], self.data['y'], c=self.data['timestamp'], cmap='viridis')
        plt.colorbar(label='Timestamp')
        plt.title('Temporal-Spatial Distribution')

        plt.tight_layout()


# 使用示例
if __name__ == "__main__":
    # 生成测试数据（人工 vs 机器）
    human_data = pd.DataFrame({
        'timestamp': np.cumsum(np.random.uniform(50, 150, 100)),  # 随机时间间隔
        'x': np.cumsum(np.random.normal(0, 2, 100)),
        'y': np.cumsum(np.random.normal(0, 2, 100))
    })

    bot_data = pd.DataFrame({
        'timestamp': np.arange(0, 5000, 50),  # 固定50ms间隔
        'x': np.linspace(0, 1000, 100),
        'y': np.sin(np.linspace(0, 4*np.pi, 100)) * 100 + 500
    })

    # 分析人工数据
    analyzer = MouseMovementAnalyzer(human_data)
    features = analyzer.extract_all_features()
    analyzer.visualize()

    # 训练异常检测模型
    # 实际应用中需要准备标注数据
    train_data = pd.concat([
        MouseMovementAnalyzer(bot_data).extract_all_features(),
        MouseMovementAnalyzer(human_data).extract_all_features()
    ], ignore_index=True)
    labels = [0, 1]  # 0:bot, 1:human

    model = IsolationForest(contamination=0.1)
    model.fit(train_data)

    # 预测新数据
    new_data = MouseMovementAnalyzer(bot_data).extract_all_features()
    prediction = model.predict(new_data)
    print("Prediction (-1=anomaly/bot, 1=normal/human):", prediction)
