import socket
import struct
import random
import signal
import sys


def run_stun_server(bind_ip='0.0.0.0', bind_port=3478):
    """
    启动一个支持Ctrl+C关闭的STUN服务器
    :param bind_ip: 绑定的IP地址（默认0.0.0.0，监听所有网卡）
    :param bind_port: 绑定的端口（默认3478，STUN标准端口）
    """
    # 创建UDP套接字（STUN基于UDP）
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except socket.error as e:
        print(f"[错误] 创建套接字失败: {e}")
        return

    # 设置套接字选项（允许端口重用，方便调试时快速重启）
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # 绑定到指定IP和端口
    try:
        sock.bind((bind_ip, bind_port))
    except socket.error as e:
        print(f"[错误] 绑定端口 {bind_port} 失败: {e}")
        sock.close()
        return

    # 设置套接字超时（1秒），避免recvfrom永久阻塞，允许响应中断信号
    sock.settimeout(1)
    print(f"STUN服务器启动成功，监听 {bind_ip}:{bind_port}... (按 Ctrl+C 关闭)")

    # 注册Ctrl+C信号处理函数（优雅退出）
    def handle_sigint(signum, frame):
        print("[提示] 收到退出信号，正在关闭服务器...")
        sock.close()  # 关闭套接字释放资源
        sys.exit(0)   # 退出程序

    signal.signal(signal.SIGINT, handle_sigint)  # 捕获Ctrl+C信号（SIGINT）
    while True:
        try:
            # 接收客户端请求（最多阻塞1秒，因套接字设置了超时）
            data, client_addr = sock.recvfrom(1024)  # 缓冲区大小1024字节
            print(f"[接收] 来自 {client_addr} 的STUN请求（长度: {len(data)} 字节）")
            # 解析STUN请求头（前20字节必须存在）
            if len(data) < 20:
                print(f"[丢弃] 无效请求：报文长度不足20字节")
                continue

            msg_type, msg_len, tid = struct.unpack('!HH16s', data[:20])
            if msg_type != 0x0001:  # 非STUN请求
                print(f"未知消息类型: 0x{msg_type:04x}")
                continue

            # 构造STUN响应（成功响应类型为0x0101）
            response = bytearray()
            # 消息头：类型0x0101，长度（后续属性长度），事务ID
            response += struct.pack('!HH16s', 0x0101, 8, tid)  # 属性长度8字节（见下文）

            # 添加XOR-MAPPED-ADDRESS属性（类型0x0020，IPv4）
            src_ip, src_port = client_addr
            src_ip_bytes = socket.inet_pton(socket.AF_INET, src_ip)  # IPv4转4字节

            # 计算XOR值（事务ID前4字节参与异或）
            tid_port = (tid[0] << 8) | tid[1]  # 事务ID前2字节作为16位端口掩码
            xor_port = src_port ^ tid_port

            tid_ip = tid[:4]  # 事务ID前4字节作为IP掩码
            xor_ip_bytes = bytes(a ^ b for a, b in zip(src_ip_bytes, tid_ip))

            # 属性格式：类型(2B) + 长度(2B) + 地址族(1B) + 保留(1B) + 端口(2B) + IP(4B)
            response += struct.pack(
                '!HHBBH4s',
                0x0020,       # 类型：XOR-MAPPED-ADDRESS
                8,            # 长度：IPv4地址占4字节
                0x01,         # 地址族：IPv4（0x01）
                0x00,         # 保留字段
                xor_port,     # 异或后的端口
                xor_ip_bytes  # 异或后的IP
            )
            print(f"[响应] 公网IP: {src_ip}, 公网端口: {src_port} -> XOR端口: {xor_port}, XOR IP: {socket.inet_ntoa(xor_ip_bytes)},length:{len(response)}")
            # 发送响应回客户端
            sock.sendto(response, client_addr)
            print(f"已向 {client_addr} 发送响应")

        except socket.timeout:
            # print("[超时] 等待服务器响应超时")
            pass

        except Exception as e:
            print(f"处理请求出错: {e}", type(e))


def stun_client(server='127.0.0.1', port=3478, timeout=5):
    try:
        # 创建UDP套接字
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)

        # 生成随机事务ID（16字节，STUN要求全局唯一）
        transaction_id = bytes(random.getrandbits(8) for _ in range(16))

        # 构造STUN请求报文（类型0x0001，无属性）
        # 消息头格式：类型(2B) + 长度(2B) + 事务ID(16B)
        request = struct.pack('!HH16s', 0x0001, 0, transaction_id)

        # 发送请求到STUN服务器
        sock.sendto(request, (server, port))
        print(f"[客户端] 已向 {server}:{port} 发送STUN请求（事务ID: {transaction_id.hex()}）")

        # 接收服务器响应
        response_data, _ = sock.recvfrom(1024)  # 最大响应长度通常不超过1024字节
        print(f"[客户端] 收到响应，长度: {len(response_data)} 字节")

        # 解析响应头（前20字节）
        if len(response_data) < 20:
            print("[错误] 响应报文长度不足（至少20字节）")
            return None, None

        # 解包消息头：类型(2B)、长度(2B)、事务ID(16B)
        msg_type, msg_len, recv_tid = struct.unpack('!HH16s', response_data[:20])

        # 检查是否为成功响应（类型0x0101表示成功响应）
        if msg_type != 0x0101:
            print(f"[错误] 非预期的消息类型: 0x{msg_type:04x}（期望0x0101）")
            return None, None

        # 检查事务ID是否匹配（防止中间人攻击或乱序包）
        if recv_tid != transaction_id:
            print("[错误] 事务ID不匹配！可能被篡改或非本次请求的响应")
            return None, None

        # 解析属性（从第20字节开始）
        attributes_start = 20
        while attributes_start < len(response_data):
            # 解析属性头：类型(2B)、长度(2B)
            attr_type, attr_len = struct.unpack('!HH', response_data[attributes_start:attributes_start+4])
            attr_start = attributes_start + 4
            attr_end = attr_start + attr_len

            # 确保属性长度不超过剩余数据长度（避免越界）
            if attr_end > len(response_data):
                print(f"[警告] 属性长度超出报文范围（类型0x{attr_type:04x}, 长度{attr_len}）")
                break

            # 提取属性值（跳过属性头）
            attr_value = response_data[attr_start:attr_end]

            # 关注关键属性：XOR-MAPPED-ADDRESS（类型0x0020，IPv4）
            if attr_type == 0x0020 and attr_len == 8:  # IPv4的XOR-MAPPED-ADDRESS长度固定为4字节
                # 属性值结构：地址族(1B) + 保留(1B) + 端口(2B) + IP(4B)
                # 注意：实际属性值的总长度是 1+1+2+4=8字节？不，STUN属性长度字段是“值部分的长度”（不包含类型和长度字段本身）
                # 正确解析：XOR-MAPPED-ADDRESS的属性值部分长度为 1（地址族） + 1（保留） + 2（端口） + 4（IP）= 8字节？
                # 但根据RFC 5389，XOR-MAPPED-ADDRESS的属性长度（attr_len）应为 4（仅IP和端口？需要重新核对协议）
                # 修正：根据RFC 5389，XOR-MAPPED-ADDRESS的属性格式为：
                #   类型: 0x0020
                #   长度: 4（对于IPv4，地址族1字节 + 保留1字节 + 端口2字节 + IP4字节，总长度是 1+1+2+4=8？但实际长度字段是“值部分的长度”，即 8-4=4？可能我之前理解有误）
                # 正确的做法是：属性长度字段（attr_len）是值部分的字节数（不包含类型和长度字段）。对于IPv4的XOR-MAPPED-ADDRESS，值部分的结构是：
                #   地址族（1字节，0x01表示IPv4）
                #   保留（1字节，必须为0）
                #   端口（2字节，XOR后的值）
                #   地址（4字节，XOR后的值）
                # 因此，值部分总长度是 1+1+2+4=8字节 → 所以attr_len应为8？
                # 这里可能之前的服务器实现有误，需要修正！

                # 重新解析（假设服务器返回的是正确的IPv4 XOR-MAPPED-ADDRESS）
                # 正确的属性值结构（8字节）：
                #   family (1B), reserved (1B), port (2B), xor_ip (4B)
                if len(attr_value) < 8:
                    print("[错误] XOR-MAPPED-ADDRESS属性值长度不足")
                    break

                family = attr_value[0]
                reserved = attr_value[1]
                xor_port = struct.unpack('!H', attr_value[2:4])[0]
                xor_ip_bytes = attr_value[4:8]

                # 计算真实的公网端口和IP（与事务ID的前部分异或）
                # 根据RFC 5389，XOR操作规则：
                #   端口 = 响应中的端口 XOR 事务ID的前2字节（作为16位整数）
                #   IP = 响应中的IP XOR 事务ID的前4字节（作为32位整数）
                # 事务ID是16字节，前2字节（16位）用于端口异或，前4字节（32位）用于IP异或

                # 提取事务ID的前2字节（端口掩码）
                tid_port_mask = transaction_id[0:2]
                tid_port = struct.unpack('!H', tid_port_mask)[0]
                real_port = xor_port ^ tid_port

                # 提取事务ID的前4字节（IP掩码）
                tid_ip_mask = transaction_id[0:4]
                tid_ip = socket.inet_ntoa(tid_ip_mask)  # 转换为点分十进制字符串（仅用于参考）
                print("事务ID前4字节（IP掩码）:", tid_ip)
                tid_ip_bytes = tid_ip_mask  # 直接使用字节掩码

                # 计算真实IP（字节异或）
                real_ip_bytes = bytes(a ^ b for a, b in zip(xor_ip_bytes, tid_ip_bytes))
                real_ip = socket.inet_ntoa(real_ip_bytes)  # 转换为点分十进制字符串

                # 检查地址族是否为IPv4（0x01）
                if family == 0x01:
                    print(f"[客户端] 公网IP: {real_ip}，公网端口: {real_port}")
                    return real_ip, real_port
                else:
                    print(f"[警告] 不支持的地址族: 0x{family:02x}（仅支持IPv4）")
                    break

            # 其他属性可扩展（如USERNAME、MESSAGE-INTEGRITY等，但客户端通常只需XOR-MAPPED-ADDRESS）
            else:
                print(f"[信息] 忽略未知属性: 类型0x{attr_type:04x}, 长度{attr_len}")

            # 移动到下一个属性
            attributes_start = attr_end

        # 如果循环结束未找到XOR-MAPPED-ADDRESS
        print("[错误] 响应中未找到XOR-MAPPED-ADDRESS属性")
        return None, None

    except socket.timeout:
        print("[错误] 接收响应超时")
        return None, None
    except Exception as e:
        print(f"[错误] 客户端异常: {e}")
        return None, None
    finally:
        sock.close()  # 确保关闭套接字


if __name__ == "__main__":
    c = input("0:service,1:client:")
    print(f"input:{c}")
    if c == "0":
        run_stun_server()
    else:
        stun_client()
