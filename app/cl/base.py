# 动态创建子类
child_class = type('Child', (custom.ICustomTask,), {})
del child_class
 