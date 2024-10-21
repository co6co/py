import os
import re
import argparse


class NetworkManage:
    _ipList: list = None
    _dev_list: list = None

    def __init__(self):
        self._dev_list = self.get_net_dev()

    def get_net_dev(self):
        """
        获取网卡名: LIST
        """
        result = os.popen('ipconfig')
        res = result.read()
        resultlist = re.findall('''(?<=以太网适配器 ).*?(?=:)|(?<=无线局域网适配器 ).*?(?=:)''', res)
        return resultlist

    def set_ip(self, dev_name: str, ip="192.168.1.99", mask="255.255.255.0", gateway="192.168.1.254", dhcp: bool = None) -> None | str:
        """
        设备IP地址
        return  
            None 设置成功
            error Message 
        """
        if dhcp:
            result = os.popen(f'netsh interface ipv4 set address name="{dev_name}" source=dhcp')
        else:
            result = os.popen(f'netsh interface ipv4 set address "{dev_name}" static {ip} {mask} {gateway}')
        res = result.read()
        return res


if __name__ == '__main__':
    manage = NetworkManage()
    print(manage._dev_list)

    parser = argparse.ArgumentParser(description="audit service.")
    group = parser.add_argument_group("获取配置")
    group.add_argument("-d", "--devices", action=argparse.BooleanOptionalAction, help="获取网络设备名称")
    group = parser.add_argument_group("配置IP")
    group.add_argument("-s", "--deviceName", help=f"设备名:<\"{'" | "'.join(manage._dev_list)}\">")
    group.add_argument("-a", "--auto", default=False, action=argparse.BooleanOptionalAction, help="自动从DHCP获取,Default:False")
    group.add_argument("-i", "--ip", default="192.168.1.99")
    group.add_argument("-m", "--mask", default="255.255.255.0")
    group.add_argument("-g", "--gateway", default="192.168.1.254")
    args = parser.parse_args()

    if args.devices:
        print(manage._dev_list)
    if args.deviceName and args.ip and args.mask and args.gateway:
        manage.set_ip(args.deviceName, args.ip, args.mask, args.gateway, dhcp=args.auto)
        print(manage._ipList)
