import asyncio
import functools
from typing import Awaitable,Type, TypeVar, Callable,Optional

T = TypeVar("T")


class TaskTimeoutError(asyncio.TimeoutError):
    pass


class TotalTimeoutError(TimeoutError):
    pass


def with_ignore(
    ignore: bool = True,
    exc_types: tuple[Type[Exception], ...] = (Exception,)
):
    def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Optional[T]:
            try:
                return await func(*args, **kwargs)
            except exc_types:
                if not ignore:
                    raise
                return None
        return wrapper
    return decorator

class AsyncLimiter:
    """异步并发限制器"""

    def __init__(self, max_concurrency: int):
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def run(self, coro: Awaitable[T]) -> T:
        async with self.semaphore:
            return await coro

    @with_ignore(True)
    async def run_timeout(
        self,
        coro: Awaitable[T],
        timeout: float = None,
        task_timeout: float = 10.0 
    ) -> T:
        """
        异步运行任务，超时返回None
        不推荐同时使用 wait_for+ timeout​
        """ 
        try:
            async with asyncio.timeout(timeout):
                async with self.semaphore:
                    try:
                        return await asyncio.wait_for(coro, timeout=task_timeout)
                    except asyncio.TimeoutError:
                        raise TaskTimeoutError(
                            f"run_timeout timeout {task_timeout} seconds"
                        )
        except TimeoutError:
            raise TotalTimeoutError(f"asyncio.timeout超时了{timeout}")
         

    def wrap(self, func: Callable[..., Awaitable[T]]):
        """
        包装异步函数，限制并发数

        示例:
        ```python
        limiter = AsyncLimiter(max_concurrency=3)
        @limiter.wrap
        async def fetch_data(i):
            await asyncio.sleep(1)
            print(i)

        async def main():
            await asyncio.gather(*[fetch_data(i) for i in range(10)])
        """

        async def wrapper(*args, **kwargs):
            async with self.semaphore:
                return await func(*args, **kwargs)

        return wrapper
