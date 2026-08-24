#!/usr/bin/env python3
"""
SOCKS5 代理服务器 - 支持用户名密码认证
仅监听 127.0.0.1，配合 SSH 隧道或云服务器使用
外网客户端 -> 云服务器:1080 -> SOCKS5 认证 -> 内网设备
"""
import asyncio
import struct
import socket

# ===== 配置 =====
LISTEN_HOST = "127.0.0.1"   # 建议只绑本地，前面套 SSH 隧道或 nginx stream SSL
LISTEN_PORT = 1080
AUTH_USERNAME = "admin"
AUTH_PASSWORD = "更换为强密码"
# ===============

async def handle_socks5_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    client_addr = writer.get_extra_info('peername')
    print(f"[+] 新连接来自: {client_addr}")

    try:
        # 1. 读取客户端支持的认证方法
        data = await reader.readexactly(2)
        version, nmethods = data[0], data[1]
        if version != 5:
            writer.close()
            return

        methods = await reader.readexactly(nmethods)

        # 2. 我们要求用户名密码认证 (0x02)
        if 0x02 not in methods:
            writer.write(struct.pack("!BB", 5, 0xFF))  # 无可接受方法
            await writer.drain()
            writer.close()
            return

        writer.write(struct.pack("!BB", 5, 0x02))  # 选择用户名密码认证
        await writer.drain()

        # 3. 用户名密码认证 (RFC 1929)
        ver = await reader.readexactly(1)
        ulen = (await reader.readexactly(1))[0]
        username = (await reader.readexactly(ulen)).decode()
        plen = (await reader.readexactly(1))[0]
        password = (await reader.readexactly(plen)).decode()

        if username != AUTH_USERNAME or password != AUTH_PASSWORD:
            writer.write(struct.pack("!BB", 1, 0x01))  # 认证失败
            await writer.drain()
            print(f"[-] 认证失败: {username}")
            writer.close()
            return

        writer.write(struct.pack("!BB", 1, 0x00))  # 认证成功
        await writer.drain()
        print(f"[+] 认证成功: {username}")

        # 4. 接收连接请求
        data = await reader.readexactly(4)
        ver, cmd, _, atyp = data
        if cmd != 0x01:  # 只支持 CONNECT
            writer.close()
            return

        # 解析目标地址
        if atyp == 0x01:  # IPv4
            addr_bytes = await reader.readexactly(4)
            target_addr = socket.inet_ntoa(addr_bytes)
        elif atyp == 0x03:  # 域名
            dlen = (await reader.readexactly(1))[0]
            target_addr = (await reader.readexactly(dlen)).decode()
        else:
            writer.close()
            return

        target_port = struct.unpack("!H", await reader.readexactly(2))[0]
        print(f"[>] 连接目标: {target_addr}:{target_port}")

        # 5. 连接目标设备
        try:
            remote_reader, remote_writer = await asyncio.open_connection(
                target_addr, target_port
            )
        except Exception as e:
            writer.write(struct.pack("!BB", 5, 0x04))  # 连接失败
            await writer.drain()
            writer.close()
            return

        # 6. 回复连接成功
        writer.write(struct.pack("!BBB", 5, 0x00, 0x00))
        writer.write(struct.pack("!B", 0x01))
        writer.write(socket.inet_aton("0.0.0.0"))
        writer.write(struct.pack("!H", 0))
        await writer.drain()

        # 7. 双向转发
        await asyncio.gather(
            forward(reader, writer, remote_writer, "客户端->目标"),
            forward(remote_reader, remote_writer, writer, "目标->客户端"),
            return_exceptions=True
        )

    except Exception as e:
        print(f"[-] 错误: {e}")
    finally:
        writer.close()

async def forward(src_reader, src_writer, dst_writer, direction):
    try:
        while True:
            data = await src_reader.read(65536)
            if not data:
                break
            dst_writer.write(data)
            await dst_writer.drain()
    except:
        pass
    finally:
        dst_writer.close()

async def main():
    server = await asyncio.start_server(handle_socks5_client, LISTEN_HOST, LISTEN_PORT)
    print(f"SOCKS5 代理启动在 {LISTEN_HOST}:{LISTEN_PORT}")
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
    """
    # 在公司内网一台 Linux 上运行上述脚本（只监听 127.0.0.1）
    # 然后用 SSH 隧道把端口映射到你的笔记本：
    ssh -N -L 1080:127.0.0.1:1080 user@gateway.company.com
    """