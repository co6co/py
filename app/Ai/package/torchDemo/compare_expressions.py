#-*- coding: utf-8 -*-
import numpy as np

# 创建一个示例Y数组
Y = np.array([-2.5, -1.0, 0.0, 1.5, 2.0])
print("原始Y数组:", Y)

# 表达式1: (Y+1).astype(int)
expr1 = (Y + 1).astype(int)
print("\n表达式1 (Y+1).astype(int):")
print("结果:", expr1)
print("值的范围:", expr1.min(), "到", expr1.max())
print("唯一值:", np.unique(expr1))

# 表达式2: ((Y>0)+1).astype(int)
expr2 = ((Y > 0) + 1).astype(int)
print("\n表达式2 ((Y>0)+1).astype(int):")
print("Y>0的布尔结果:", (Y > 0))
print("+1后的结果:", (Y > 0) + 1)
print("最终结果:", expr2)
print("值的范围:", expr2.min(), "到", expr2.max())
print("唯一值:", np.unique(expr2))

# 解释vis.scatter的Y参数要求
print("\n=== vis.scatter函数中Y参数的要求 ===")
print("1. Y不是y轴坐标，而是数据点的类别标签")
print("2. Y必须是与X行数相同长度的整数数组")
print("3. 每个整数代表一个类别，通常从1开始")
print("4. 不同的整数会用不同颜色显示")
print("\n为什么表达式2更适合vis.scatter:")
print("- 表达式1产生的类别太多（每个不同的值都是一个类别）")
print("- 表达式2明确将数据分为两类，适合分类可视化")
print("- 类别数量与legend数量一致时，可视化效果最佳")