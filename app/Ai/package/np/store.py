import numpy as np
from pathlib import Path
my_arr1 = np.arange(0, 16).reshape((4, 4))
rootPath = Path('D:\\temp')
save_path = rootPath/'my_arr.npy'
# 存储为 my_arr1.data.npy 文件
print("save时是可以没有.npy后缀的")
np.save(save_path, my_arr1)
print("load 时必须有 .npy后缀")
my_arr1 = np.load(save_path)
print("保存后读取的数据\n->> \n", my_arr1)

my_arr1 = np.arange(16).reshape((4, 4))
my_arr2 = np.random.randn(16).reshape((4, 4))
save_path = rootPath/'my_arr2'
np.savez(save_path, my_arr1=my_arr1, my_arr2=my_arr2)
print("多个数组的后缀名为 .npz ")
arch = np.load(f"{save_path}.npz")
print("保存后读取的数据\n->> \n", arch)

my_arr1 = arch['my_arr1']
my_arr2 = arch['my_arr2']
print("保存后读取的数据2\n->> \n", my_arr1)
print("保存后读取的数据2\n->> \n", my_arr2)
save_path = rootPath/'my_arr2_by_compressed'
np.savez_compressed(save_path, my_arr1=my_arr1, my_arr2=my_arr2)
arch = np.load(f"{save_path}.npz")
print("从压缩文件中读取\n->> \n", arch)
for key in arch:
    print("key->", key)
    print("data->", arch[key])


# 其他的操作
my_arr1 = np.arange(6)
my_arr2 = my_arr1
print("my_arr2 = my_arr1不会创建新的数组对象，只是多了一个引用")
print("my_arr2 is my_arr1", my_arr2 is my_arr1)

# 视图
my_arr2 = my_arr1.view()
print("my_arr2 = my_arr1.view()会创建新的数组对象，但是数据完全一致，两个数组共享数据")
print("my_arr2 is my_arr1", my_arr2 is my_arr1)
print("my_arr2.base is my_arr1", my_arr2.base is my_arr1)
print("my_arr2.flags.owndata", my_arr2.flags.owndata)
print("my_arr2 的形状发生改变时，my_arr1 的形状并不会发生改变。但是，当 my_arr2 的数据发生改变时，my_arr1 会发生同样的改变")
my_arr1 = np.arange(6)
print("my_arr1 原数组->", my_arr1)
my_slice = my_arr1[2:4]

my_slice[:] = 10
print("改变切片的所有值为10,原数组->", my_arr1, '切片->', my_slice)
print("视图除了数据是共享的，其他都是独立的。另外，前面介绍的**切片操作**，得到的也是一个数组的视图")

# 拷贝
my_arr1 = np.arange(6)
my_arr2 = my_arr1.copy()

print(my_arr2 is my_arr1)
print(my_arr2.base is my_arr1)
print("my_arr1.copy(),my_arr1 和 my_arr2 是两个独立的数组.")


# 广播 broadcasting
# 指的是不同形状的数组之间的算术运算的执行方式

my_arr1 = np.arange(6)
my_arr2 = my_arr1 * 6
print("在这个[my_arr1 * 6]乘法运算中，标量值 6 被广播到了其他所有的元素上")
print("通过减去列平均值的方式对数组的每一列进行距平化处理")
my_arr1 = np.random.randn(4, 3)
print("my_arr1 原数组->", my_arr1)
my_arr1_mean = my_arr1.mean(0)
print("my_arr1_mean->", my_arr1_mean)
demeaned = my_arr1 - my_arr1_mean
print("距平化处理->", demeaned)
