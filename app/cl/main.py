# -*- coding:utf-8 -*-

import sys
from co6co.utils import find_files, convert_size,  convert_to_bytes, split_value_unit
from co6co.utils import log
from co6co.utils.singleton import Singleton
import os
from datetime import datetime
print("当前文件路径：", __file__, )
print('__name__:', __name__)
log.start_mark("本文件代码")
with open(__file__, 'r', encoding='utf-8') as file:
    list = file.readlines()
    print("文件代码：\n", *list)
log.end_mark("本文件代码")


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


# 检查是否在虚拟环境中
if hasattr(sys, 'real_prefix'):
    print("real_prefix:", sys.prefix)
if hasattr(sys, 'base_prefix'):
    print("base_prefix:", sys.base_prefix)
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print("虚拟环境路径:", sys.prefix)
else:
    print("未使用虚拟环境，当前 Python 路径:", sys.prefix)
