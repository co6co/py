from co6co_task.service import CustomTask
from services.cfService import CfService
from sanic import Sanic
from co6co.utils import log
import asyncio
from co6co.utils import try_except
import socket
from co6co.utils import http
from co6co_sanic_ext import sanics


class CfTaskMgr(CustomTask.ICustomTask):
    name = "更新cf解析记录"
    code = "CfTaskMgr"

    def __init__(self ):
        super().__init__()
        self.cfService: CfService = asyncio.run(CfService.instance())
        runonce = self.cfService.item.get("runOnce", True)

        if runonce:
            log.succ("cfTaskMgr:", "启动", "执行一次...")
            self.main()
        pass

    def get_ipv4_address(self):
        try:
            # 向提供 IP 查询服务的网站发送请求
            response = http.get('https://icanhazip.com/')
            # 检查响应状态码
            if response.status_code == 200:
                return response.text
            else:
                print(f"请求失败，状态码: {response.status_code}")
        except Exception as e:
            print(f"请求发生错误: {e}")
        return None

    def get_ipv6_address(self):
        try:
            # 获取主机名
            hostname = socket.gethostname()
            # 获取主机的所有地址信息
            addr_info = socket.getaddrinfo(hostname, None, socket.AF_INET6)
            for info in addr_info:
                # 提取 IPv6 地址
                ipv6_address = info[4][0]
                if ipv6_address != '::1':  # 排除本地回环地址
                    return ipv6_address
            return None
        except Exception as e:
            print(f"发生错误: {e}")
            return None

    @try_except
    def main(self):
        # self.cfService.update_dns_record()
        item = self.filter_records()
        type = self.cfService.item.get("type")
        ip = self.get_ipv6_address() if type == "AAAA" else self.get_ipv4_address()
        if not ip:
            log.warn("cfTaskMgr:", "获取IP失败")
            return
        result = None
        name = self.cfService.item.get("name")
        if not item:
            result = self.cfService.create_dns_record(type, name, content=ip, ttl=256)
        else:
            item["content"] = self.get_ipv6_address()
            data = {
                "record_id": item.get("id"),
                "type": item.get("type"),
                "name": name,
                "content": ip,
                "ttl": item.get("ttl"),
                "proxied": item.get("proxied"),
                "comment": item.get("comment")
            }

            result = self.cfService.update_dns_record(**data)
            log.info(f"更新{name}:", result)
            pass

    def get_dns_record(self, record_id):
        return self.cfService.get_dns_record(record_id)

    @try_except
    def filter_records(self):
        all_records = self.cfService.list_dns_records()
        success = all_records.get("success")
        if success:
            list = all_records.get("result")
        name = self.cfService.item.get("name")
        for item in list:
            if item.get("name") == name:
                return item
        return None
