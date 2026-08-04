#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
网络魔术字开机（Wake-on-LAN）
发送魔术包唤醒局域网内的远程设备
"""

import socket
import struct
import binascii


def create_magic_packet(mac_address: str) -> bytes:
    """
    构造 WoL 魔术包
    
    魔术包格式：
    - 前 6 字节：全 0xFF（同步头）
    - 后跟 16 次重复的 MAC 地址（共 96 字节）
    - 总计 102 字节
    """
    # 清理 MAC 地址，只保留十六进制字符
    mac = mac_address.replace(':', '').replace('-', '').replace('.', '')
    
    if len(mac) != 12:
        raise ValueError(f"无效的 MAC 地址: {mac_address}")
    
    # 将 MAC 地址转为 6 个字节
    mac_bytes = binascii.unhexlify(mac)
    
    # 构造魔术包：6 字节 0xFF + 16 次 MAC 地址
    magic_header = b'\xff' * 6
    magic_payload = mac_bytes * 16
    magic_packet = magic_header + magic_payload
    
    return magic_packet


def wake_on_lan(mac_address: str, broadcast_addr: str = '255.255.255.255', port: int = 9):
    """
    发送魔术包唤醒目标设备
    
    :param mac_address: 目标设备的 MAC 地址，格式如 '00:11:22:33:44:55'
    :param broadcast_addr: 广播地址，默认全网广播
    :param port: 端口号，常用 7 或 9
    """
    packet = create_magic_packet(mac_address)
    
    # 创建 UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    
    try:
        sock.sendto(packet, (broadcast_addr, port))
        print(f"✅ 魔术包已发送到 {broadcast_addr}:{port}")
        print(f"   目标 MAC: {mac_address}")
        print(f"   包大小: {len(packet)} 字节")
    except Exception as e:
        print(f"❌ 发送失败: {e}")
    finally:
        sock.close()


def wake_multiple_devices(devices: list):
    """批量唤醒多台设备"""
    for device in devices:
        mac = device.get('mac')
        broadcast = device.get('broadcast', '255.255.255.255')
        port = device.get('port', 9)
        
        if mac:
            wake_on_lan(mac, broadcast, port)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='网络魔术字开机工具 (Wake-on-LAN)')
    parser.add_argument('mac', nargs='?', help='目标 MAC 地址')
    parser.add_argument('-b', '--broadcast', default='255.255.255.255', help='广播地址')
    parser.add_argument('-p', '--port', type=int, default=9, help='端口号 (默认 9)')
    parser.add_argument('-f', '--file', help='从文件读取 MAC 地址列表批量唤醒')
    
    args = parser.parse_args()
    
    if args.file:
        # 从文件批量读取
        devices = []
        with open(args.file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    devices.append({
                        'mac': parts[0],
                        'broadcast': parts[1] if len(parts) > 1 else '255.255.255.255',
                        'port': int(parts[2]) if len(parts) > 2 else 9
                    })
        wake_multiple_devices(devices)
    
    elif args.mac:
        wake_on_lan(args.mac, args.broadcast, args.port)
    
    else:
        # 交互模式
        print("=== 网络魔术字开机工具 ===\n")
        mac = input("请输入目标 MAC 地址: ").strip()
        broadcast = input("请输入广播地址 (回车默认 255.255.255.255): ").strip() or '255.255.255.255'
        
        if mac:
            wake_on_lan(mac, broadcast)
        else:
            print("未输入 MAC 地址，退出。")