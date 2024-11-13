from typing import Generator
from co6co.utils import log


def echo_generator() -> Generator[None, str, bool]:
    """
    当生成器结束时，会引发 StopIteration 异常，可以通过 StopIteration 异常的 value 属性获取最终返回值： 
        return 语句返回一个最终值
    """
    received = yield
    print(f"Received: {received}")
    return True


def generator() -> Generator[int, str, bool]:
    """
    每执行一次 next 和 send 都会带走一个值
    """
    for i in range(0, 10):
        received = yield i
        print(f"Received: {received}")
    return True


# 创建生成器对象
gen = echo_generator()
# 初始化生成器
next(gen)  # 必须先调用一次 next() 或 send(None)
# gen.send(None)
# 发送值给生成器
try:
    gen.send("Hello")
except StopIteration as e:
    print("返回的值：", e.value)
log.start_mark("第二个生成器标记")
gen = generator()
gen.send(None)
try:
    for i in gen:
        d = gen.send(f"你好{i}")
        print(d)
except StopIteration as e:
    print("返回的值：", e.value)
