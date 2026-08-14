#
# 自己想的逻辑AI进行了优化
# 未进行调试
#
from socket import socket, timeout as SocketTimeout, error as SocketError
import argparse
import time
import threading
import signal


def connect_server(ip, port, stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            s = socket()
            s.settimeout(5)
            s.connect((ip, port))
            s.settimeout(None)
            return s
        except (ConnectionRefusedError, OSError, SocketError):
            time.sleep(10)
    return None


def handle(ip, port, stop_event: threading.Event):
    s = connect_server(ip, port, stop_event)
    if s is None:
        return
    try:
        while not stop_event.is_set():
            try:
                s.settimeout(1)
                data = s.recv(1024)
                if not data:
                    break
                print("recv", data)
                s.send(data)
            except SocketTimeout:
                continue
    except OSError:
        pass
    finally:
        s.close()
        print("Connection closed.")


def main(args):
    """
    主函数
    """
    stop_event = threading.Event()

    def signal_handler(signum, frame):
        print("\nShutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal_handler)

    t = threading.Thread(target=handle, args=(args.ip, args.port, stop_event))
    t.start()
    try:
        while t.is_alive():
            t.join(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        stop_event.set()

    t.join(5)
    print("Exit.")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--ip", type=str, default="127.0.0.1")
    args.add_argument("--port", type=int, default=8080)
    args = args.parse_args()
    main(args)
