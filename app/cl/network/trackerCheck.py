import requests
import socket

from urllib.parse import urlparse
import argparse
from co6co.task import ThreadTask
from typing import Tuple, IO, Dict
from co6co.utils import log
import urllib.parse
from co6co.utils import hash
import random


import struct


class UDPTrackClient:
    @staticmethod
    def create_connection_request():
        # 协议 ID (固定值)
        protocol_id = 0x41727101980
        # 动作类型 (连接请求为 0)
        action = 0
        # 随机事务 ID
        transaction_id = random.randint(0, 2**32 - 1)
        # 打包数据
        data = struct.pack('>QII', protocol_id, action, transaction_id)
        return data, transaction_id

    @staticmethod
    def parse_connection_response(data):
        action, transaction_id, connection_id = struct.unpack('>IIQ', data)
        return action, transaction_id, connection_id

    @staticmethod
    def create_announce_request(connection_id, info_hash, peer_id):
        # 动作类型 (宣布请求为 1)
        action = 1
        # 随机事务 ID
        transaction_id = random.randint(0, 2**32 - 1)
        # 已下载字节数
        downloaded = 0
        # 剩余字节数
        left = 1000000
        # 已上传字节数
        uploaded = 0
        # 事件类型 (0: 无事件, 1: 开始, 2: 完成, 3: 停止)
        event = 0
        # IP 地址 (0 表示使用发送方的 IP)
        ip = 0
        # 密钥 (随机值)
        key = random.randint(0, 2**32 - 1)
        # 期望的对等方数量 (-1 表示默认值)
        num_want = -1
        # 监听端口
        port = 6881

        data = struct.pack('>QII20s20sQQQIIIiH',
                           connection_id, action, transaction_id,
                           info_hash, peer_id,
                           downloaded, left, uploaded,
                           event, ip, key, num_want, port)
        return data, transaction_id

    @staticmethod
    def parse_announce_response(data):
        action, transaction_id = struct.unpack('>II', data[:8])
        interval = struct.unpack('>I', data[8:12])[0]
        leechers = struct.unpack('>I', data[12:16])[0]
        seeders = struct.unpack('>I', data[16:20])[0]
        peers = []
        for i in range(20, len(data), 6):
            ip_bytes = data[i:i + 4]
            ip = socket.inet_ntoa(ip_bytes)
            port = struct.unpack('>H', data[i + 4:i + 6])[0]
            peers.append((ip, port))
        return action, transaction_id, interval, leechers, seeders, peers

    def query(self, tracker_address: Tuple):
        # 种子文件的 info_hash (示例值，需替换为实际值)
        info_hash = b'\x12\x34\x56\x78\x90\x12\x34\x56\x78\x90\x12\x34\x56\x78\x90\x12\x34\x56\x78\x90'
        # 客户端的 peer_id (示例值，需替换为实际值)
        peer_id = b'-PC1234-ABCDEFGHIJKL'

        # 创建 UDP 套接字
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(5)

        try:
            # 发送连接请求
            connection_request, transaction_id = self. create_connection_request()
            sock.sendto(connection_request, tracker_address)

            # 接收连接响应
            try:
                data, server = sock.recvfrom(4096)
                action, received_transaction_id, connection_id = self. parse_connection_response(data)
                if action == 0 and received_transaction_id == transaction_id:
                    print(f"成功获取连接 ID: {connection_id}")
                else:
                    print("连接响应无效")
                    return
            except socket.timeout:
                print("连接请求超时")
                return

            # 发送宣布请求
            announce_request, transaction_id = self. create_announce_request(connection_id, info_hash, peer_id)
            sock.sendto(announce_request, tracker_address)

            # 接收宣布响应
            try:
                data, server = sock.recvfrom(4096)
                action, received_transaction_id, interval, leechers, seeders, peers = self. parse_announce_response(data)
                if action == 1 and received_transaction_id == transaction_id:
                    print(f"间隔时间: {interval} 秒")
                    print(f"下载者数量: {leechers}")
                    print(f"种子数量: {seeders}")
                    print("可用对等方:")
                    for peer in peers:
                        print(peer)
                    return {"interval": interval, "leechers": leechers, "seeders": seeders, "peers": peers}
                else:
                    print("宣布响应无效")
            except socket.timeout:
                print("宣布请求超时")
        finally:
            sock.close()
        return None


def getClientParam():
    # 模拟客户端信息
    client_id = hash.md5("ABC123"+str(random.random()))[0:20]
    torrent_info_hash = "SHA1_HASH_OF_TORRENT"
    client_ip = "127.0.0.1"
    client_port = 6881
    downloaded = 0
    left = 1000000  # 剩余字节数
    uploaded = 0
    event = "started"

    params = {
        "info_hash": torrent_info_hash,
        "peer_id": client_id,
        "port": client_port,
        "uploaded": uploaded,
        "downloaded": downloaded,
        "left": left,
        "event": event,
        "ip": client_ip
    }
    return params


def check_http_tracker(tracker_url, *, clientCategory: int = 0):
    try:
        params = getClientParam()
        query_string = urllib.parse.urlencode(params)
        full_url = f"{tracker_url}?{query_string}"
        response = requests.get(full_url, timeout=5)
        if not response.status_code == 200:
            log.info(f"接受request到{tracker_url}->{response.status_code},", response.content)
            return False
        content = response.content.decode('utf-8')
        if "peer id" in content:
            log.info(f"接受request到{tracker_url}->{response.status_code},", content)
            return True
    except requests.RequestException as e:
        # log.err(f"解析：{tracker_url}error", e)
        log.warn(f"{tracker_url}->error:{e}")
        return False


def check_udp_tracker(tracker_address):
    client = UDPTrackClient()
    data = client.query(tracker_address)
    if data:
        log.info(f"{tracker_address}->", data)
        return True
    return False


def parse_url(url):
    parsed = urlparse(url)
    protocol = parsed.scheme
    host = parsed.hostname
    port = parsed.port
    # print(f"{url}->{protocol},{host},{port}")
    return protocol, host, port


def generate_task(filePath, outPath, encoding, param: dict):
    with open(filePath, "r", encoding=encoding) as file, open(outPath, 'w', encoding=encoding) as file2:
        for line in file:
            line = line.strip()
            if line:
                yield line, file2, param
            else:
                continue


def handler(param: Tuple[str, IO, Dict]):
    line, file2, all = param
    pro, host, port = parse_url(line)
    if pro == 'udp':
        result = check_udp_tracker((host, port))
    else:
        result = check_http_tracker(line, **all)
    if result:
        log.succ(f"{line} 有效")
        file2.write(line + "\n")
        file2.flush()
    else:
        log.warn(f"{line} 无效")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MD5")
    parser.add_argument("-f", "--file", default='./trackers.txt', help="trackers 所在文本文件")
    parser.add_argument("-o", "--out", default='./trackers_valid.txt', type=str, help=f"输出路径")
    parser.add_argument("-e", "--encoding", default='utf-8', type=str, help=f"文本编码")
    parser.add_argument("-m", "--maxThreads", default=8, type=int, help="处理线程数")
    parser.add_argument("-u", "--url", default=None, type=str, help="测试URL")
    httpParser = parser.add_argument_group("http解析")
    httpParser.add_argument("-c", "--clientCategory", default=1, type=int, help="0:chrome,1:firefox")

    args = parser.parse_args()
    outPath = args.out
    # 获取参数组中的参数名
    group_param_names = [action.dest for action in httpParser._group_actions]

    # 提取参数组中的参数
    group_params = {name: getattr(args, name) for name in group_param_names}
    if args.url:
        # content = get_dynamic_text_content(args.url, args.clientCategory)
        # log.warn(f"{args.url}->", content)
        # log.warn(args.__dict__, group_params)
        with open(outPath, 'w') as file2:
            handler((args.url, file2, group_params))
        parser.print_help()

        exit(0)
    else:
        encoding = args.encoding
        filePath = args.file
        outPath = args.out
        maxThreads = args.maxThreads
        outData = []
        task = ThreadTask(handler, generate_task(filePath, outPath, encoding, group_params))
        task.start(maxThreads)
