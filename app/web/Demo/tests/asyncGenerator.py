import asyncio,random

async def async_gen():
    for i in range(3):
        try:
            await asyncio.sleep(random.uniform(0.1, 0.5))
            yield i
        except asyncio.CancelledError:
            #通常不推荐在生成器内部捕获CancelledError并继续，因为取消操作意味着任务应该被终止。
            # 如果你确实需要这样做，请确保生成器的状态一致性，并且有另外的消费者来继续迭代生成器，
            # 否则生成器可能会一直挂起，造成资源泄漏。

            # 捕获取消异常，然后继续循环，尝试下一个yield
            print(f"生成器在yield {i}处被取消，继续下一个")
            continue # 这里常见的做法为 抛出异常 而不是继续循环
        except Exception as e:
            print(f"发生错误: {e}")
            raise

async def main():
    agen = async_gen()
    print("#"*8,"async for 使用方式1","#"*8)
    # 使用方式1 async for
    async for item in agen:
        print(item)
    print("#"*32)

    print("#"*8,"anext使用方式2","#"*8)
    # 使用方式2 anext() 函数
    agen=async_gen()
    while True: 
        try:
            item = await anext(agen)
            print(item)
        except StopAsyncIteration:
            print("迭代结束")
            break
    print("#"*32)
    print("#"*8,"手动 __anext__","#"*8)
    agen = async_gen() 
    while True:
        try:
            #item = await agen.__anext__() 
            
            #当我们使用 asyncio.wait_for时，如果超时发生，wait_for会取消内部的任务（或可等待对象）。
            # 对于 agen.__anext__()来说，取消操作可能会在生成器内部产生一个 asyncio.CancelledError，
            # 这可能会导致生成器的状态变得不一致，甚至可能使生成器提前结束，而不会抛出 StopAsyncIteration。
            #异步生成器中 最佳实践是在生成器内部适当处理CancelledError，或者在生成器外部通过aclose()方法来关闭生成器。

            # 如果不显式处理asyncio.CancelledError，那么当任务被取消时，生成器会收到一个CancelledError异常。这个异常会从当前挂起的await点抛出
            # 如果生成器内部没有捕获这个异常，那么生成器会在抛出CancelledError的地方停止，并且生成器的状态会被保持，但生成器不会继续执行，而是被取消。

            # 生成器其需要处理  asyncio.CancelledError 异常时需要的操作
            # 否则可能导致agen.__anext__() 与 anext(agen) 调用时结果不一致

            item = await asyncio.wait_for(agen.__anext__(), timeout=0.3) 
            #item = await asyncio.wait_for(anext(agen), timeout=0.3) 
            print(item)
        except asyncio.TimeoutError:
            print("获取下一个元素超时")
        except StopAsyncIteration:
            print("结束")
            break
        except Exception as e:
            print(f"发生错误: {e}") 
asyncio.run(main())