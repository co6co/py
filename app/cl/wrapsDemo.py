from functools import wraps
import inspect


def exector(use: bool = False):  # 接受参数的装饰器工厂
    def decorator(f):  # 实际的装饰器
        """
        @wraps
            1. 保留原始函数的名称：确保装饰后的函数仍然显示为原始函数的名称。
            2. 保留原始函数的文档字符串：确保装饰后的函数仍然具有原始函数的文档字符串。
            3. 保留原始函数的参数签名：确保装饰后的函数的参数签名与原始函数一致。

        *args:int --> [Tuple[int,...]
        **kwargs --> Dict[str, float]
        """
        @wraps(f)
        def decorated_function(*args, **kwargs):
            print("kwargs:", kwargs)
            print("args:", args)
            if use:
                print("Using the executor...")
                # 在这里可以添加额外的功能逻辑
            return f(*args, **kwargs)  # 调用原始函数并返回结果
        return decorated_function
    return decorator


def exector2(f):  # 简单装饰器
    @wraps(f)
    def decorated_function(*args, **kwargs):
        '''
        特性               @exector()	                             @exector
        是否支持参数       支持，通过括号传递参数	                   不支持，行为固定
        实现方式           装饰器工厂函数	                          简单装饰器
        调用过程           先调用装饰器工厂函数，返回实际装饰器	        直接将目标函数传递给装饰器
        灵活性             更灵活，可以根据参数调整装饰器行为	        行为固定，无法动态调整
        装饰器的嵌套        可以在不同的函数上多次使用	                只能在一个函数上使用一次
        '''
        # 在这里可以添加额外的功能逻辑
        return f(*args, **kwargs)  # 调用原始函数并返回结果
    return decorated_function


@exector(use=False)  # 使用参数化的装饰器
def my_function(x, y):
    """这是一个简单的加法函数"""
    return x + y


@exector()
def my_function2(x, y):
    """这是一个简单的加法函数"""
    return x - y


@exector2
def my_function3(x, y):
    """这是一个简单的加法函数"""
    return x - y


result = my_function(3, 5)
result = my_function2(3, 5)
result = my_function3(3, 5)
print(result)

# 查看函数的元信息
print(my_function.__name__)
print(my_function.__doc__)
print(inspect.signature(my_function))

'''
没有@wraps:
decorated_function
None
(*args, **kwargs)

有@wraps:
my_function
这是一个简单的加法函数
(x, y)
'''
