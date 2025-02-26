# -*- coding:utf-8 -*-

from co6co.utils import find_files, convert_size,  convert_to_bytes, split_value_unit
from co6co.utils import log
from co6co.utils.singleton import Singleton
import os
from datetime import datetime
print(__file__, __name__)
with open(__file__, 'r', encoding='utf-8') as file:
    list = file.readlines()
    print("文件代码：\n", *list)


def filterFile(fileName: str):
    ignore = ['.vmdk']  # 该文件比较大一般不会重复
    # 可能返回 ''
    _, ext = os.path.splitext(fileName)
    if ext in ignore:
        return False
    return True


s1 = Singleton()
s2 = Singleton()
print(s1 == s2, id(s1), id(s2))  # 输出: True
print(s1.createTime)
print(s2.createTime)
