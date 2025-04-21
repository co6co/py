import socket


def check_port(host, port) -> tuple[bool, str]:
    """
    检查指定主机的指定端口是否开放。
    :param host: 主机名或IP地址。
    :param port: 端口号。
    :return: 一个元组，第一个元素为True表示开放，False表示不开放，第二个元素为错误信息或None。
    """
    try:
        # 创建一个 TCP 套接字
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 设置超时时间
        sock.settimeout(2)
        # 尝试连接到指定的主机和端口
        result = sock.connect_ex((host, port))
        if result == 0:
            return True, None
        else:
            return False, "端口不开发"

    except socket.gaierror as e:
        return False, e
    except socket.error as e:
        return False, e
    finally:
        # 关闭套接字
        sock.close()


if __name__ == "__main__":
    # 请替换为你要检测的主机和端口
    target_host = "192.168.300.101"
    target_port = 554
    print(*(target_host, target_port), 'is', check_port(target_host, target_port))
