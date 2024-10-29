
# -*- encoding:utf-8 -*-
# 泛型测试
'''
1. Generic 的类绑定
    绑定会的检查，本身编译器不报错
2. Type[_S] 类类型 
    class_ 与 type(class_.__name__,(class_),{}) 的区别

'''
from typing import Generic, TypeVar, Type, Any, Callable


# 创建类型变量
# 用于泛型函数、类或方法中
# 限制其只能绑定到 Session 类或其子类
_S = TypeVar("_S", bound="Session")


class Session:
    sessionName = None

    @staticmethod
    def static():
        print("静态方法")

    def __init__(self, name) -> None:
        print("{}__init__".format(self.__class__))
        self.sessionName = name
        pass

    def print(self):
        print("hellow:", self.sessionName)

# Generic 是一个基类，用于定义泛型类。当你创建一个继承自 Generic 的类时，
# 你需要指定一个或多个类型变量，这些变量将在类的方法和属性中被使用。


class sessiom_factory(Generic[_S]):
    '''
    确保 只能处理符合特定条件的会话对象，从而提高代码的类型安全性和可维护性。

    接受一个类型为类型为 _S 的会话对象。
    由于 _S 被限制为 Session 或其子类

    _S 被限制为 Session 类或其子类。但是，如果你尝试传递一个不是 Session 类或其子类的实例，
    Python 解释器本身不会报错，因为 Python 是动态类型语言。只有在运行时才会出现问题。

    解释器中不会报错，但它违反了你在 sessiom_factory 类中设定的类型约束。
    解决方法1:
        使用静态类型检查工具：确保在开发过程中使用静态类型检查工具（如 mypy）,这样可以在编写代码时发现类型错误
    解决方法2:
        运行时检查：构造函数中添加运行时类型检查，确保传入的参数符合预期类型。

    '''
    class_: Type[_S]

    def __init__(self, class_: Type[_S] = Session) -> Callable[..., _S]:
        print("{}__init__".format(self.__class__))
        # if not isinstance(session, Session):
        if not issubclass(class_, Session):
            raise TypeError(f"{class_} 不是 {Session} 的子类")
        self.class_ = class_  # 直接引用已存在的类对象。你不需要创建新的类，而是直接使用现有的类。
        self.class_ = type(class_.__name__, (class_,), {})  # 是 _class 的子类

        """
        type 用于 动态创建类，
        这个新类继承自 class_，并且没有任何额外的属性或方法。
        """
        print("--", self.class_.__name__)
        super().__init__()

    def __call__(self, *args, **local_kw: Any) -> _S:
        return self.class_(*args, **local_kw)


class CustomSession(Session):
    """
    具体的Session类
    """

    def __init__(self, name: str, user_id):
        super().__init__(name)
        self.user_id = user_id

    def print(self):
        print("Custom", self.sessionName)


class demo:
    """
    创建个不是Session子类的类
    """

    def print(self):
        print("ccc")


S = sessiom_factory[CustomSession](class_=CustomSession)
s = S(123, 456)
s.print()

# 报错 有类型检查
S = sessiom_factory[demo](class_=demo)
s = S()
s.print()
