import os
import re
from co6co.utils.File import File
x = {}
for data in File.readBytes("C:\\Users\\Administrator\\Downloads\\未命名.json", 1):
    s = data.hex()
    r = r'^\d+$'
    if re.match(r, s):
        n = int(s)
        if n <= 33 and n > 0:
            value = x.get(n)
            if value == None:
                x.update({n: 1})
            else:
                x.update({n: value+1})
sorted_x = sorted(x, key=lambda k: (x[k]), reverse=True)  # (x[k], k)
print(sorted_x)
sorted_xx = [{i: x[i]} for i in sorted_x]
print(sorted_xx)
