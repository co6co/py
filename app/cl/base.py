# 动态创建子类

def abc(a,/,b):
    print(a)
    print(b)


aaa=(1,2)
print(type(aaa),aaa)
abc(*aaa)

child_class = type('Child', (custom.ICustomTask,), {})
del child_class
 