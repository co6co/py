import numpy as np

my_arr1 = np.random.random(12)
print(my_arr1)
# 12 个元素的一维数组，
# 将 my_arr1 重塑成一个 3 行 4 列的二维数组
# my_arr1 并没有改变
my_arr2 = my_arr1.reshape(3, 4)
print("新矩阵:reshape(3, 4)", my_arr2, "old Id->", id(my_arr1), "new Id->", id(my_arr2))

my_arr1.shape = (3, 4)  # 改变原数组的形状
print("改变原数组的形状,shape=(3,4)", my_arr1, id(my_arr1))
print("old Id->", id(my_arr1))
my_arr1 = my_arr1.ravel()  # 重新变为 一维数组
print("创建新数组 变为 一维矩阵 ravel()", my_arr1)
# 一维数组组合
print("一维数组组合....")
my_arr1 = np.arange(0, 3)
my_arr2 = np.arange(3, 6)
my_arr3 = np.arange(6, 9)
print("一维数组组合:row_stack,column_stack,hstack,vstack")
print(np.row_stack((my_arr1, my_arr2, my_arr3)), "row_stack行组合", '\n')
print(np.column_stack((my_arr1, my_arr2, my_arr3)), "column_stack列组合", '\n')
print(np.hstack((my_arr1, my_arr2, my_arr3)), "行hstack组合", '\n')
print(np.vstack((my_arr1, my_arr2, my_arr3)), "列vstack组合", '\n')

# 组合二维数组
print("组合二维数组....")
my_arr1 = np.ones((3, 3))
my_arr2 = np.zeros((4, 3))

print(f"组合 -垂直方向 vstack【列数数需要相等】->\n{my_arr1}\n+\n{my_arr2}=\n", np.vstack((my_arr1, my_arr2)))
print("-垂直方向,另一种实现:np.row_stack((my_arr1, my_arr2))", np.row_stack((my_arr1, my_arr2)))
my_arr1 = np.ones((3, 4))
my_arr2 = np.zeros((3, 5))

print(f"\n组合 -水平方向 hstack【行数需要相等】->\n{my_arr1}\n+{my_arr2}=\n", np.hstack((my_arr1, my_arr2)))
print("-垂直方向,另一种实现:np.column_stack((my_arr1, my_arr2))", np.column_stack((my_arr1, my_arr2)))
# 三维数组
# my_arr1 = np.ones((3, 3, 3))
# print(my_arr1)

# 拆分
my_arr1 = np.arange(0, 16).reshape((4, 4))
print("拆分....")
print(my_arr1)
print("拆垂直方向:np.vsplit(arr,2)->\n", np.vsplit(my_arr1, 2))  # 垂直方向
print("拆水平拆分方向:np.hsplit(arr,2)->\n", np.hsplit(my_arr1, 2))  # 水平方向

print("\n指定拆分位置：np.split(my_arr1, [1, 3], axis=1)->\n", np.split(my_arr1, [1, 3], axis=1))
print("指定拆分位置：np.split(my_arr1, [1, 3], axis=0)->\n", np.split(my_arr1, [1, 3], axis=0))
