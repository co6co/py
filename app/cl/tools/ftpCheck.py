import argparse
import ftplib
import os
from co6co.task.pools import limitThreadPoolExecutor
from concurrent.futures import Future, CancelledError
from co6co.utils import log
import signal


def connect_to_ftp(host, user, password):
    try:
        ftp = None
        print(f"尝试连接到 {host} 使用 {user}/{password}...")
        # 创建 FTP 对象
        ftp = ftplib.FTP(host)
        # 登录 FTP 服务器
        ftp.login(user, password)
        print(f"成功连接到 {host}")
        # 打印当前工作目录
        # log.info("当前工作目录:", ftp.pwd())
        return True
    except ftplib.all_errors as e:
        print(f"连接失败: {e}")
        return None
    finally:
        # 关闭 FTP 连接
        if ftp:
            ftp.quit()


def result(f: Future):
    """
    当密码正确后后面的线程还来不及退出时  f.exception 不出错
    """
    try:
        index, result_list, server, user, pwd, futureList = f.args
        futureList: list[Future] = futureList
        result = f.result()
        futureList.remove(f)
        result_list[index].update({"host": server})
        if result:
            result_list[index].get("userPwd").append({"user": user, "pwd": pwd})
    except CancelledError:
        print(f"任务{server}已取消。")
    except Exception as e:
        print(f"任务{server}执行出错: {e}")


def check(servers, users, pwds, max_workers: int = 4):
    pool = limitThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ftpCheck")
    # 定义信号处理函数
    futureList: list[Future] = []
    canelFlag = False

    def signal_handler(signum, frame):
        print(f"尝试取消任务...")
        nonlocal canelFlag
        canelFlag = True
        for f in futureList:
            re = f.cancel()

    # 注册信号处理函数
    signal.signal(signal.SIGINT, signal_handler)

    result_list = [{"userPwd": []}] * len(servers)  # 列表中的都指向同一个对象   即所有的{"userPwd": []} 为统一对象
    result_list = [{"userPwd": []} for _ in servers]
    for index, server in enumerate(servers):

        server = server.strip()
        for user in users:
            user = user.strip()
            for pwd in pwds:
                if canelFlag:
                    break
                pwd = pwd.strip()
                f = pool.submit(connect_to_ftp, server, user, pwd)
                futureList.append(f)
                f.args = (index, result_list, server, user, pwd, futureList)
                f.add_done_callback(result)

    pool.shutdown(wait=True)
    print("检测完成.")
    return result_list


if __name__ == "__main__":
    # nuitka --standalone --onefile --windows-icon-from-ico=c:\Users\Administrator\Pictures\cardano.ico .\tools\ftpCheck.py
    parser = argparse.ArgumentParser(description="检测ftp服务器是否存在指定的弱口令")
    parser.add_argument("-s", "--serversFile", default='./servers.txt',  type=str, help="服务器列表文件,每行一个服务器地址,default:./servers.txt")
    parser.add_argument("-u", "--userFile", default="./user.txt", type=str, help="用户名列表文件,每行一个用户名,default:./user.txt")
    parser.add_argument("-p", "--pwdFile", default="./pwd.txt", type=str, help="用户名密码文件,每行一个用户名密码,default:./pwd.txt")
    parser.add_argument("-m", "--max_workers", default=4, type=int, help="线程数")

    args = parser.parse_args()
    serversFile = args.serversFile
    userFile = args.userFile
    pwdFile = args.pwdFile
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
        data = check(servers, users, pwds, max_workers=max_workers)
        print("*"*50)
        print("服务器\t用户名/密码\t用户名/密码\t.../...")
        for d in data:
            if len(d.get("userPwd")) > 0:
                print(d.get("host"), *[i.get("user")+"/"+i.get("pwd") for i in d.get("userPwd")], sep="\t")
            else:
                print(d.get("host"), "无弱口令", sep="\t")
        print("*"*50)
