import sys
import argparse
import ftplib
import os
from co6co.task.pools import limitThreadPoolExecutor
from co6co.task import ThreadTask
from concurrent.futures import Future, CancelledError
import signal
import tempfile
import time
import subprocess
from co6co.utils.log import progress_bar
import threading


def connect_to_ftp(host, user, password):
    try:
        ftp = None
        print(f"尝试连接到 {host} 使用 {user}/{password}...")
        # 创建 FTP 对象
        ftp = ftplib.FTP(host, timeout=10)
        # 登录 FTP 服务器
        ftp.login(user, password)
        # print(f"成功连接到 {host}")
        # 打印当前工作目录
        # log.info("当前工作目录:", ftp.pwd())
        return True
    except ftplib.all_errors as e:
        # print(f"连接失败: {e}")
        return False
    except Exception as e:
        # print(f"发生错误: {e}")
        return False
    finally:
        # 关闭 FTP 连接
        if ftp:
            ftp.quit()


def generator(servers: list, users: list, pwds: list):
    for index, server in enumerate(servers):
        for user in users:
            for pwd in pwds:
                if canelFlag:
                    return
                yield (index, server.strip(), user.strip(), pwd.strip())


canelFlag = False


def signal_handler(signum, frame):
    print(f"尝试取消任务...")
    global canelFlag
    canelFlag = True


def check2(servers: list, users: list, pwds: list, max_workers: int = 4):
    result_list = [{"host": _.strip(), "userPwd": [], "checking": True, "usePwdCount": len(users)*len(pwds)} for _ in servers]

    times = len(result_list)*len(users)*len(pwds)

    def hander(param: tuple):
        try:
            index, server, user, pwd = param
            result = connect_to_ftp(server, user, pwd)
            if result:
                print("登录成功", server, user, pwd)
                result_list[index].get("userPwd").append({"user": user, "pwd": pwd})
        except Exception as e:
            # print(f"任务{server}执行出错: {e}")
            pass
        finally:
            count = result_list[index].get("usePwdCount")
            count = count-1
            # print(f"检测结果: {count},{result_list[index]}")
            result_list[index].update({"usePwdCount": count})

    threads = ThreadTask(hander, generator(servers, users, pwds))
    threading.Thread(target=threads.start, daemon=True, args=(max_workers,)).start()

    while True:
        time.sleep(1)
        if canelFlag:
            print("任务提前结束，检查结果可能不全...")
            threads.stop()
            break
        _count = sum(item.get("usePwdCount") for item in result_list if item.get("usePwdCount") > 0)
        if _count == 0:
            break
        progress_bar((times-_count) / times, title="检测进度")

    return result_list


def get_executable_dir():
    return os.path.dirname(os.path.abspath(sys.executable))
    # if getattr(sys, 'frozen', False):
    #    # 如果是打包后的单文件程序，返回临时解压目录
    #    return sys._MEIPASS
    # else:
    #    # 如果是普通的脚本，返回脚本所在目录
    #    return os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    # nuitka --standalone --onefile --windows-icon-from-ico=c:\Users\Administrator\Pictures\cardano.ico .\tools\ftpCheck.py
    parser = argparse.ArgumentParser(description="检测ftp服务器是否存在指定的弱口令")
    curr = get_executable_dir()
    parser.add_argument("-s", "--servers_file", default=f"{curr}\\servers.txt",  type=str, help=f"服务器列表文件, 每行一个服务器地址, default: {curr}\\servers.txt")
    parser.add_argument("-u", "--user_file", default=f"{curr}\\user.txt", type=str, help=f"用户名列表文件,每行一个用户名,default:{curr}\\user.txt")
    parser.add_argument("-p", "--pwd_file", default=f"{curr}\\pwd.txt", type=str, help=f"用户名密码文件,每行一个用户名密码,default:{curr}\\pwd.txt")
    parser.add_argument("-m", "--max_workers", default=4, type=int, help="线程数")

    args = parser.parse_args()

    serversFile = args.servers_file
    userFile = args.user_file
    pwdFile = args.pwd_file
    print("输入的参数:\nip ->", serversFile, "\nuser ->", userFile, "\npwd ->", pwdFile, "\n线程数 ->", args.max_workers)
    max_workers = args.max_workers
    showHelp = False
    while True:
        if os.path.exists(serversFile) == False or os.path.exists(userFile) == False or os.path.exists(pwdFile) == False:
            print("检查传入的文件是否都存在！")
            showHelp = True
            break
        with open(serversFile, 'r') as file:
            servers = file.readlines()
        with open(userFile, 'r') as file:
            users = file.readlines()
        with open(pwdFile, 'r') as file:
            pwds = file.readlines()
        if len(servers) == 0:
            print("serversFile 为空")
            showHelp = True
        if len(users) == 0 or len(pwds) == 0 or len(servers) == 0:
            print("检查用户名、密码、服务器文件是否为空！")
            showHelp = True
        break
    if showHelp:
        parser.print_help()
    else:
        # 注册信号处理函数
        signal.signal(signal.SIGINT, signal_handler)
        data = check2(servers, users, pwds, max_workers=max_workers)
        print("*"*50)
        dbFolder = tempfile.gettempdir()
        tempFile = None
        with tempfile.TemporaryFile(dir=dbFolder, mode="w+", delete=False, suffix=".txt") as file:
            print("临时文件路径:", file.name)
            if canelFlag:
                print("任务提前结束，检查结果可能不全...", file=file, flush=True)
            print("服务器\t用户名/密码\t用户名/密码\t.../...")
            tempFile = file.name
            # 向临时文件中写入数据
            # temp.write(b'Hello World!')
            # 将文件指针移到开始处
            # temp.seek(0)
            # 读取数据
            for d in data:
                tmp = []
                tmp.append(d.get("host"))
                if len(d.get("userPwd")) > 0:
                    tmp.append([i.get("user")+"/"+i.get("pwd") for i in d.get("userPwd")])
                    print(*tmp, sep="\t", file=file, flush=True)
                    print(*tmp, sep="\t")
                else:
                    tmp.append('无弱口令')
                    print(*tmp, sep="\t", file=file, flush=True)
                    print(*tmp, sep="\t")
        process = subprocess.Popen(["notepad", file.name])
        print("*"*50)
