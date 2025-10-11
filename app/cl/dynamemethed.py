import types


class Request:
    def __init__(self, name="Default"):
        self.name = name

    def echo(self):
        print("000")


# 在对象上新增方法
################ 方案1########################
# 创建实例
r = Request()

# 定义要动态添加的方法


def dynamic_print(self, *args):
    print("Dynamic print:", *args)


# 将方法绑定到实例r
r.print = types.MethodType(dynamic_print, r)

# 调用新方法
r.print("Hello, World!")  # 输出: Dynamic print: Hello, World!

################ 方案2 ########setattr################
r = Request()

# 直接通过 setattr 动态添加
setattr(r, 'print', types.MethodType(lambda self, *args: print("方案2使用setattr:", *args), r))
r.print("Test")  # 输出: Dynamic: Test
################ 方案3 ########类上添加################


@classmethod
def class_print(cls, *args):
    # 定义类方法
    print(cls, id(cls))
    print("方案3Class method:", *args)


# 绑定到类
"""
缺点是类方法，不是对象方法
"""
Request.print = class_print

# 所有实例均可调用
r = Request()
print(r, id(r))
r.print("Global")  # 输出: Class method: Global

############## 方案4 ########使用闭包（捕获上下文）################
# 方案1的另类用法
r = Request("Instance-A")
# 通过闭包捕获实例属性


def create_print_method(instance: Request):
    def print_method(self: Request, *msg: str):
        print(f"闭包捕获上下文: {id(instance)} vs {id(self)}")
        print(f"[{instance.name}] {msg}")
    return print_method


r.print = types.MethodType(create_print_method(r), r)
r.print("Context-aware", "Hello")  # 输出: [Instance-A] Context-aware Hello
