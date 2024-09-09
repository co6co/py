import os


def child_process(read_fd):
    # 从管道中读取数据
    with os.fdopen(read_fd, 'r') as pipe_read:
        data = pipe_read.read()
        print(f"Child received: {data}")


def parent_process():
    # 创建管道
    read_fd, write_fd = os.pipe()

    # 创建子进程 # Windows没有提供fork系统调用
    # 它是当前进程（父进程）的一个副本。子进程和父进程共享相同的程序代码和打开的文件描述符，但它们有不同的进程ID (PID) 和各自的内存空间。
    # 当os.fork() 被调用时，它会在父进程和子进程中返回不同的值：
    pid = os.fork()
    # 调用之后，程序会分别在父进程和子进程中执行相应的分支代码

    if pid == 0:
        # 在子进程中关闭写端
        os.close(write_fd)
        child_process(read_fd)
    else:
        # 在父进程中关闭读端
        os.close(read_fd)

        # 向管道写入数据
        with os.fdopen(write_fd, 'w') as pipe_write:
            pipe_write.write("Hello from parent!")

        # 等待子进程结束
        os.waitpid(pid, 0)


parent_process()
