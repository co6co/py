import numpy as np
import sys

arrayList = [1, 33, ]


def leanSize(arrayList):
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
print("列表对象本身（不包含元素）的大小为 56 个字节，每增加一个元素，元素本身的大小为 28 个字节，指向元素的指针大小为 8 个字节")
leanSize(["1"])
leanSize(["1", "33"])
leanSize(["1", "33", "6333333333333333333333333", ])
