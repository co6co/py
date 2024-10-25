# -*- encoding:utf-8 -*-
# 修饰类
def class_decorator(cls):
    cls.new_attribute = "New Attribute"
    return cls


def method_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Calling method: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper


def static_method_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Calling static method: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper


def class_method_decorator(func):
    def wrapper(cls, *args, **kwargs):
        print(f"Calling class method: {func.__name__}")
        return func(cls, *args, **kwargs)
    return wrapper


@class_decorator
class MyClass:
    def __new__(cls, *args, **kvg):
        """
        静态方法，用于创建类的新实例。
        它是对象实例化过程的第一步， 负责分配内存并返回一个空的对象。
        __new__ 方法必须返回一个实例对象，否则不会调用 __init__ 方法 

        """
        print("__new__")
        # 避免显示调用父类【父对象】的类[对象]方法
        return super(MyClass, cls).__new__(cls)

    def __init__(self, value):
        print("__init__", value)
        self.value = value

    @method_decorator
    def display(self):
        print(f"Value: {self.value}")

    @staticmethod
    @static_method_decorator
    def static_method():
        print("This is a static method")

    @classmethod
    @class_method_decorator
    def class_method(cls):
        print("This is a class method")


class SubClass(MyClass):
    def __new__(cls, *args, **kvg):
        """

        """
        print("SUB,__new__", args, kvg)
        # 避免显示调用父类【父对象】的类[对象]方法
        return super(SubClass, cls).__new__(cls)

    def __init__(self, d):
        print("SUB,__init__", d)
        # python 3 后能自动推动出来  ==>super(SubClass,self).__init__(d)
        super().__init__(d)


# 创建类的实例并访问新属性
obj = MyClass(10)
print(obj.new_attribute)  # 输出: New Attribute

# 调用装饰的实例方法
obj.display()  # 输出: Calling method: display
#        Value: 10

# 调用装饰的静态方法
MyClass.static_method()  # 输出: Calling static method: static_method
#        This is a static method

# 调用装饰的类方法
MyClass.class_method()  # 输出: Calling class method: class_method
#        This is a class method


subObj = SubClass(12)
subObj.display()
