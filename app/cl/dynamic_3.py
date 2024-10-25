# 示例源代码
source_code = """
class Demo:
    def __init__(self):
        self.name="属性名"
    def print(self):
        print(self.name)
    def set(self,name:str):
        self.name=name
def hello(name):
    return f'Hello, {name}!'
# 这里代码只要编译就会执行
print(hello('World'))

# 检查局部变量
if 'x' in locals():
    print("局部变量 x 已定义",x)
else:
    print("局部变量 x 未定义")

# 检查全局变量
if 'x' in globals():
    print("全局变量 x 已定义",x)
else:
    print("全局变量 x 未定义")
"""


# 编译源代码
compiled_code = compile(source_code, '<string>', 'exec')

# 执行编译后的代码
exec(compiled_code)

# 定义全局变量和局部变量
global_vars = {"x": 100}
local_vars = {"x": 99}

# 执行编译后的代码，并指定全局和局部变量
# 两参数是本地变量和全局变量为同一字典
exec(compiled_code, global_vars, local_vars)
# 访问局部变量中的结果


print("全局变量：", global_vars)
print("局部变量",  local_vars)
# 1. 调用方法
for a in global_vars:
    print(a, global_vars[a])
result = global_vars['hello']('Python')
print("globel", result)
result = local_vars['hello']('Python')
print(result)  # 输出: Hello, Python!

# 2. 创建对象及调用对象方法
demo = local_vars['Demo']()
demo.print()
demo.set("改变值")
demo.print()
