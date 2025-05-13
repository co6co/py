import numpy as np
import sys
import time


# 1. 占用内存
def leanSize(arrayList):
    """
    列表对象本身（不包含元素）的大小为 56 个字节，每增加一个元素，元素本身的大小为 28 个字节，指向元素的指针大小为 8 个字节
    """
    listSize = sys.getsizeof(arrayList)
    elementSize = 0
    elementsSize = 0
    if len(arrayList) > 0:
        elementSize = sys.getsizeof(arrayList[0])
        elementsSize = len(arrayList)*sys.getsizeof(arrayList[0])
    print("arrayList->", arrayList, "的大小：")
    print("listSize: ", listSize)
    print("elementSize: ", elementSize)
    print(f"elementsSize: {len(arrayList)}*{elementSize}=", elementsSize)
    print("listSize Total Size:[listSize+elementsSize] ", listSize + elementsSize)


print("sys.getsizeof(1): ", sys.getsizeof(1))
leanSize([])
leanSize([1])
leanSize([1, 33])
leanSize([1, 33, 6, ])


def leanSizeNumpy(npArray):
    """
    ndarray 对象的大小包含了元素在内。不包含任何元素的 ndarray 对象的大小为 96 字节，每增加一个元素，增加 8 个字节
    """
    print("npArray->", npArray, "的大小：")

    print("Size without the size of the elements: ", sys.getsizeof(npArray))
    print(f"Size of all the elements: {npArray.itemsize} * {npArray.size}=", npArray.itemsize * npArray.size)


my_arr1 = np.array([])
print("元素类型: ", my_arr1.dtype)
leanSizeNumpy(my_arr1)
my_arr1 = np.array([1])
print("元素类型: ", my_arr1.dtype)
leanSizeNumpy(my_arr1)
my_arr1 = np.array([1, 2])
leanSizeNumpy(my_arr1)
leanSizeNumpy(np.array([1, 2, 3]))

print("----------------------------")
# 2. 更快的运行速度

size = 10000000
my_list1 = range(size)
my_list2 = range(size)
begin_list_time = time.time()
result_list = [(a * b) for a, b in zip(my_list1, my_list2)]

end_list_time = time.time()
list_cost_time = end_list_time - begin_list_time

print("Time taken by Lists to perform multiplication:", list_cost_time, "seconds")

my_arr1 = np.arange(size)
my_arr2 = np.arange(size)
begin_arr_time = time.time()
result_arr = my_arr1 * my_arr2
end_arr_time = time.time()
arr_cost_time = end_arr_time - begin_arr_time
print("Time taken by NumPy Arrays to perform multiplication:", arr_cost_time, "seconds")
print("Numpy in this example is " + str(list_cost_time / arr_cost_time) + "faster!")

# 3. 更方便的运行
# 在原对象上进行运算
my_arr1 = np.arange(10)
print(my_arr1, "对象ID->", id(my_arr1), type(my_arr1))
my_arr1 += 1
print("加1:",my_arr1, "对象ID->", id(my_arr1))
my_arr1 //= 2
print("对2整除:", my_arr1)
my_arr1 %= 3
print("对2取余:", my_arr1)
