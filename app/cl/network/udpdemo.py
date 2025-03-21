import socket
from co6co.utils import log


def udpConnect(server_address: dict):
    try:
        # 创建 UDP 套接字
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 设置超时时间
        sock.settimeout(5)
        # 尝试发送一个简单的消息
        message = b'Hello, Tracker!'
        sock.sendto(message, server_address)
        # 尝试接收响应
        data, _ = sock.recvfrom(1024)
        log.info(f"接受到{server_address}->", data.decode())
        return bool(data)
    except socket.timeout:
        return False
    except Exception:
        return False
    finally:
        sock.close()
