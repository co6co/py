import numpy as np

my_arr = np.arange(0, 10)
print(my_arr[0])
print(my_arr[1])
print(my_arr[-1]) # 倒数第一个元素
print(my_arr[-3]) # 倒数第三个元素
ll=my_arr[[2,4,6,-2,-4,-1]] #新生成一个ndarry
ll.shape=(2,3) 
print(ll,type(ll),id(ll),id(my_arr))
print(my_arr[[2,4,6,-2,-4]]) #  
# 二维数组索引
my_arr = np.arange(1, 10).reshape(3,3)
print("二维数组：",my_arr)
print("第一行的第二个元素:",my_arr[0, 1]) #传入行和列
print("第二行的第三个元素是：",my_arr[1, 2])
print("第二行、第三行的第一个元素:",my_arr[[1,2], 0]) 
print("第二行、第三行的第三个元素:",my_arr[[1,2], 2]) 

# 布尔索引
my_arr = np.random.random(9)
print(my_arr)
my_arr = my_arr < 0.5
print("小于0.5的的列表：", my_arr,type(my_arr),my_arr.dtype  )
names = np.array(['Bob','Joe','Will','Bob'])
print(names == 'Bob')
print(names[names == 'Bob'])

#
my_arr = np.random.random((4, 4))
names = np.array(['Bob','Joe','Will','Bob'])
print("原始数组：",my_arr,names == 'Bob',my_arr[names == 'Bob'],sep="\n")

# 切片
my_arr = np.arange(9)
print(my_arr[1:5])
print(my_arr[:])
print(my_arr[1:8:2])
print(my_arr[::2])
print(my_arr[:5:2])
print(my_arr[:8:])
print(my_arr[::-1])

my_arr = np.arange(9).reshape((3,3))
print(my_arr[:])
print(my_arr[0, :])
print(my_arr[:, 0])
print(my_arr[:2, :1])
print(my_arr[0:2, 0:2])
print(my_arr[[0, 2], :])
print(my_arr[::-1])

# 遍历
my_arr = np.arange(9).reshape((3,3))
for r in my_arr:
    for c in r:
        print(c) 
for i in my_arr.flatten():
    print(i)
# 行优先
for i in np.nditer(my_arr, order='C'):
    print(i)
# 列优先
for i in np.nditer(my_arr, order='F'):
    print(i)
