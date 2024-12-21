import copy
from co6co.enums import Base_Enum, Base_EC_Enum


class categoryDesc:
    select: int = None
    dan: int = None
    z: int = None
    b: int = None

    def __init__(self):
        self.select = 0
        self.dan = 0
        self.z = 0
        self.b = 0


class category(Base_EC_Enum):
    """
    用户类别
    """
    zx_10_0_7_6 = "zx_10_0_7_6",  "矩阵_10_0_7_6", 0
    zx_15_0_7_5 = "zx_15_0_7_5", "矩阵_15_0_7_5", 1
    zx_10_1_7_6 = "zx_10_1_7_6", "矩阵_10_1_7_6", 2

    def toDesc(self):
        key: str = self.getKey()
        arr = key.split("_")
        result = categoryDesc()
        result.select = arr[1]
        result.dan = arr[2]
        result.z = arr[3]
        result.b = arr[4]
        return result


arr_10_7_6 = [
    [1, 2, 3, 5, 6, 9, 10],
    [3, 5, 6, 7, 8, 9, 10],
    [2, 3, 4, 5, 6, 9, 10],
    [1, 3, 4, 5, 6, 9, 10],
    [1, 2, 4, 5, 6, 7, 8],
    [1, 2, 3, 4, 5, 7, 8],
    [1, 2, 4, 7, 8, 9, 10],
    [1, 2, 3, 4, 6, 7, 8]
]
arr_15_7_5 = [
    [1, 5, 7, 8, 10, 11, 12],
    [2, 4, 7, 8, 9, 10, 12],
    [2, 3, 5, 10, 12, 13, 14],
    [1, 2, 6, 10, 11, 12, 15],
    [3, 6, 9, 10, 12, 13, 15],
    [1, 3, 8, 9, 10, 14, 15],
    [5, 6, 8, 9, 10, 11, 13],
    [1, 2, 3, 4, 7, 10, 14],
    [3, 4, 6, 7, 10, 11, 14],
    [1, 4, 5, 7, 10, 13, 15],
    [2, 4, 5, 7, 9, 10, 11],
    [4, 5, 8, 11, 12, 14, 15],
    [1, 4, 6, 8, 9, 11, 12],
    [1, 2, 3, 5, 8, 9, 12],
    [1, 3, 4, 6, 8, 12, 13],
    [1, 5, 6, 7, 9, 12, 14],
    [3, 7, 9, 11, 12, 13, 15],
    [2, 6, 7, 8, 13, 14, 15],
    [3, 4, 5, 6, 7, 8, 14],
    [2, 3, 7, 8, 11, 13, 15],
    [1, 2, 3, 5, 6, 11, 14],
    [1, 2, 4, 9, 11, 13, 14],
    [1, 2, 4, 5, 6, 7, 13],
    [2, 3, 4, 5, 6, 9, 15]
]
arr_10_1_7_6 = [
    [1, 2, 3, 4, 8, 9, "10*"],
    [1, 5, 6, 7, 8, 9, "10*"],
    [1, 2, 3, 4, 5, 6, "10*"],
    [1, 2, 3, 4, 5, 7, "10*"],
    [1, 2, 3, 4, 6, 7, "10*"],
    [3, 5, 6, 7, 8, 9, "10*"],
    [2, 4, 5, 6, 7, 8, "10*"]
]


def padding(lst: list, arr: list, *dan: int):
    tmpArr = copy.deepcopy(arr)
    for ar in tmpArr:
        flag = 0
        for index, a in enumerate(ar):
            if "*" in str(a):
                ar[index] = dan[flag]
                flag += 1
                continue
            for index2, val in enumerate(lst):
                if a == (index2+1):
                    ar[index] = val
                    break
    return tmpArr


'''
测试

result = padding([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], arr_15_7_5, 30)
print("源数组", arr_15_7_5)
print("目标数组", result)

'''