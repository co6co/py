
from sanic import Request
from co6co_sanic_ext.model.res.result import Result 

from co6co_web_db.view_model import BaseMethodView 

from view_model.tools import data 


class View(BaseMethodView):
    async def get(self, request: Request):
        """
        获取网络配置
        """
        # 获取互联网IP地址
        # 可能经过多个nginx代理，需要从X-Forwarded-For头中获取
        ip = request.headers.get("X-Forwarded-For", request.ip)
        # 如果X-Forwarded-For包含多个IP，取第一个
        if ip and "," in ip:
            ip = ip.split(",")[0].strip()
        
        # 构建完整的网络配置信息
        network_config = {
            "ip": ip,
            "client_ip": request.ip,
            "user_agent": request.headers.get("User-Agent", ""),
            "referer": request.headers.get("Referer", ""), 
        }
        
        return Result.success(data=network_config)
