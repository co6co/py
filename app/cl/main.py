# -*- coding:utf-8 -*-
from co6co.utils import log
import os
print(__file__, __name__)
with open(__file__, 'r', encoding='utf-8') as file:
    list = file.readlines()
    print("文件代码：\n", *list)
