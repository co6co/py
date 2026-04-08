import asyncio
import sys


async def test_subprocess():
    loop = asyncio.get_event_loop()
    print(loop)
    # 在Windows上，应该是ProactorEventLoop
    # 然后尝试创建子进程
    process = await asyncio.create_subprocess_exec(
        sys.executable, '--version',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    print(stdout.decode())

if __name__ == '__main__':
    asyncio.run(test_subprocess())
