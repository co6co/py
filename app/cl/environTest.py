import os
from functools import reduce
envs = os.environ.copy()
max_value = max(envs.keys(), key=len)
max_length = max(len(s) for s in envs.keys())
print("最长的字符:{},长度：{}".format(max_value, max_length))
max_value = reduce(lambda x, y: x if len(x) > len(y) else y, envs.keys())
print("最长的字符", max_value)

for index, key in enumerate(envs):
    print('{:0>{}}'.format(index, 3), key.rjust(max_length, ' '), "-->", "\r\n\t".join(envs[key] .split(";")))
