from functools import wraps
from sanic import Blueprint, Sanic
from co6co.utils import log
from sanic.request import Request
import inspect


def ConfigEntryExample(f):
    """
    缓存配置相关 - 改进版，支持从位置参数或关键字参数获取code
    """
    @wraps(f)
    async def _function(*args, **kwargs):
        cacheManage = None
        code = None
        
        # 1. 查找Request对象
        for arg in args:
            if isinstance(arg, Request):
                # 这里假设您有一个ConfigCache类可以处理Request
                cacheManage = "ConfigCache实例"  # 示例，实际应替换为真实的ConfigCache(arg)
                break
        
        # 2. 获取函数签名，分析code参数的位置
        sig = inspect.signature(f)
        param_names = list(sig.parameters.keys())
        
        # 3. 从关键字参数中获取code
        if "code" in kwargs:
            code = kwargs["code"]
        else:
            # 4. 从位置参数中获取code
            # 遍历参数名，查找"code"参数的位置
            if "code" in param_names:
                code_index = param_names.index("code")
                # 确保位置参数数量足够，并且该位置的参数不是self/cls
                if len(args) > code_index and param_names[code_index] == "code":
                    code = args[code_index]
        
        # 5. 执行原函数
        value = await f(*args, **kwargs)
        
        # 6. 处理缓存逻辑
        if code is not None and "SYS_CONFIG" in code:
            if cacheManage is not None:
                log.err(f"设置配置项'{code}'为'{value}'")
                # cacheManage.setConfig(code, value)  # 示例，实际应替换为真实的缓存设置逻辑
            else:
                log.warn("cacheManage 未找到 Request 参数")
        
        return value
    
    return _function


# 示例使用
class ExampleService:
    @ConfigEntryExample
    async def method1(self, request, code, value):
        """通过位置参数传递code"""
        return f"Method1 result: {value}"
    
    @ConfigEntryExample
    async def method2(self, code="default", value=None):
        """通过关键字参数传递code"""
        return f"Method2 result: {value}"
    
    @ConfigEntryExample
    async def method3(self, request, other_param, code, value):
        """code在位置参数中间"""
        return f"Method3 result: {value}"


# 测试代码（示例）
async def test():
    service = ExampleService()
    
    # 模拟Request对象
    mock_request = "MockRequest"  # 实际应替换为真实的Request对象
    
    # 测试位置参数调用
    result1 = await service.method1(mock_request, "SYS_CONFIG_TEST", {"key": "value"})
    print(f"Test 1 result: {result1}")
    
    # 测试关键字参数调用
    result2 = await service.method2(code="SYS_CONFIG_TEST", value={"key": "value"})
    print(f"Test 2 result: {result2}")
    
    # 测试code在位置参数中间的情况
    result3 = await service.method3(mock_request, "other", "SYS_CONFIG_TEST", {"key": "value"})
    print(f"Test 3 result: {result3}")


# 如果直接运行此文件，可以测试装饰器
if __name__ == "__main__":
    import asyncio
    asyncio.run(test())