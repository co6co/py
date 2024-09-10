import subprocess

# 定义要执行的命令
command = ['python', '-c', 'import sys; print(sys.stdin.read().upper())']

# 创建子进程
process = subprocess.Popen(
    command,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True  # 使用文本模式（Python 3）
)

# 定义要发送给子进程的数据
input_data = "hello, world!"

try:
    # 使用 communicate 发送数据，并获取输出
    output, error = process.communicate(input=input_data, timeout=5)
except subprocess.TimeoutExpired:
    # 如果子进程在 5 秒内没有完成，则终止子进程
    process.kill()
    output, error = process.communicate()
    print("Process did not finish within the timeout.")

# 输出结果
print("Output:", output.strip())
print("Error:", error.strip())

# 检查子进程的退出码
exit_code = process.returncode
print("Exit Code:", exit_code)
