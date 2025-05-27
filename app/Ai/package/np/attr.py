import numpy as np
import platform
print(platform.architecture())  # 输出 ('64bit', 'WindowsPE')，如果安装32位Python，输出 ('32bit', 'WindowsPE')
print("np配置...", "*"*20)
print(np.__config__.show())  # 查看NumPy配置信息
print("np配置.", "*"*20)
print(np.iinfo(np.int_).bits)


def pring_attr(my_arr: np.ndarray):
    print("元素类型：", my_arr.dtype)
    print("数组大小:", my_arr.size)
    print("数组维度:", my_arr.ndim)
    print("各个维度大小:", my_arr.shape)
    print("元素占用空间:", my_arr.itemsize)
    print()


one_arr = np.array([1, 2, 3])
pring_attr(one_arr)
two_arr = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
pring_attr(two_arr)

# 创建
my_arr = np.arange(0, 12).reshape(3, 4)
print("my_arr: ", my_arr)
my_arr = np.arange(0, 54).reshape(6, 9)
print("my_arr: ", my_arr)
# 线性空间上均匀分布
print("线性空间上均匀分布:np.linspace(0, 10,21)..reshape(3, 7)->\n", np.linspace(0, 10, 21).reshape(3, 7), '\n')
print("zeros(6)-> ",  np.zeros(6), "\nzeros((2,3))->", np.zeros((2, 3)), '\n')
print("np.ones((3,2))->\n", np.ones((3, 2)), '\n')
print("np.random.random((3,2))->\n", np.random.random((3, 2)), '\n')


# 运算
my_list1 = [1, 2, 3, 4, 5, 6]
my_list2 = [7, 8, 9, 10, 11, 12]
my_arr1 = np.array(my_list1)
my_arr2 = np.array(my_list2)
print(f"{my_arr1}+{my_arr2}={my_arr1+my_arr2}")
print(f"{my_arr1}-{my_arr2}={my_arr1-my_arr2}")
print(f"{my_arr1}*{my_arr2}={my_arr1*my_arr2}")
print(f"{my_arr1}/{my_arr2}={my_arr1/my_arr2}")

my_arr1 = np.arange(1, 10).reshape(3, 3)
my_arr2 = np.arange(2, 11).reshape(3, 3)
print(f"{my_arr1}+{my_arr2}={my_arr1+my_arr2}\n")
print(f"{my_arr1}-{my_arr2}={my_arr1-my_arr2}\n")
print(f"{my_arr1}*{my_arr2}={my_arr1*my_arr2}\n")
print(f"{my_arr1}/{my_arr2}={my_arr1/my_arr2}\n")

# 点乘
# 只有当矩阵 A 的列数（column）等于矩阵 B 的行数（row）时，A 与 B 才可以相乘
my_arr1 = np.arange(1, 10).reshape(3, 3)
my_arr2 = np.ones((3, 3))
dot_arr = np.dot(my_arr1, my_arr2)
print(f"{my_arr1}.dot({my_arr2})={dot_arr}\n")
print("**"*20)
my_arr = np.arange(1, 7).reshape(2, 3)
print(f"对{my_arr}")
print("开方", np.sqrt(my_arr))
print("log", np.log(my_arr))
print("sin", np.sin(my_arr))
print("cos", np.cos(my_arr))
print("max", np.max(my_arr))
print("max 0", np.max(my_arr, axis=0))
print("max 1", np.max(my_arr, axis=1))
print("sum 0", np.sum(my_arr, axis=0))
print("min 0", np.min(my_arr, axis=0))


print("mean 0", np.mean(my_arr, axis=0))
print("std标准差 0", np.std(my_arr, axis=0))
print("cumsum累加和 0", np.cumsum(my_arr, axis=0))
print("cumprod累计积 0", np.cumprod(my_arr, axis=0))
