
from sanic import Request
from co6co.data.result import Result 

from co6co_sanic_ext.view_model import BaseClsView 

from view_model.tools import data 


class View(BaseClsView):
    async def get(self):
        """
        获取网络配置
        """
        # 获取互联网IP地址
        # 可能经过多个nginx代理，需要从X-Forwarded-For头中获取
        ip = self.request.headers.get("X-Forwarded-For", self.request.ip)
        # 如果X-Forwarded-For包含多个IP，取第一个
        if ip and "," in ip:
            ip = ip.split(",")[0].strip()
        
        # 构建完整的网络配置信息
        network_config = {
            "ip": ip,
            "client_ip": self.request.ip,
            "user_agent": self.request.headers.get("User-Agent", ""),
            "referer": self.request.headers.get("Referer", ""), 
        }
        
        return Result.success(data=network_config)
