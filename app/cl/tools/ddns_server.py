import asyncio
import json
import time
from pathlib import Path
from aiohttp import web

# ==================== 配置项 ====================
CONFIG = {
    "PORT": 8888,                # 服务端口
    "HOST": "0.0.0.0",           # 监听所有网卡
    "TOKEN": "my-ddns-token",   # 安全密钥（设备上报必须带）
    "DB_FILE": "ddns_data.json", # 数据存储文件
}
# ===============================================

class DDNSService:
    def __init__(self):
        self.db_file = Path(CONFIG["DB_FILE"])
        self.records = {}  # 结构: { "域名": {"ip": "x.x.x.x", "update_time": 时间戳} }
        self.load_data()   # 启动时加载历史数据

    def load_data(self):
        """从文件加载 DDNS 记录"""
        if self.db_file.exists():
            try:
                with open(self.db_file, "r", encoding="utf-8") as f:
                    self.records = json.load(f)
            except Exception as e:
                print(f"加载 DDNS 记录时出错: {e}")
                self.records = {}

    def save_data(self):
        """保存 DDNS 记录到文件"""
        with open(self.db_file, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

    async def update_ip(self, domain: str, ip: str):
        """更新域名对应的 IP"""
        self.records[domain] = {
            "ip": ip,
            "update_time": int(time.time())
        }
        # 异步保存（不阻塞主线程）
        await asyncio.to_thread(self.save_data)
        return True

    def get_ip(self, domain: str):
        """查询域名 IP"""
        return self.records.get(domain, {}).get("ip")

# 全局单例
ddns_service = DDNSService()

# ==================== 路由处理 ====================
# 未经测试，主要需要要测试 ddns 客户端，请求使用那些参数和路径 
async def handle_update(request):
    """
    设备上报 IP 接口
    请求示例: GET /update?domain=home.abc.com&token=xxx&ip=1.2.3.4
    不传 ip 则自动使用请求来源 IP
    """
    # 获取参数
    domain = request.query.get("domain", "").strip()
    token = request.query.get("token", "").strip()
    client_ip = request.query.get("ip") or request.remote  # 优先使用传入的IP

    # 鉴权
    if token != CONFIG["TOKEN"]:
        return web.json_response({"code": 403, "msg": "无效的TOKEN"})

    # 校验域名
    if not domain:
        return web.json_response({"code": 400, "msg": "缺少domain参数"})

    # 更新IP
    await ddns_service.update_ip(domain, client_ip)
    return web.json_response({
        "code": 200,
        "msg": "更新成功",
        "domain": domain,
        "ip": client_ip
    })

async def handle_query(request):
    """查询域名对应IP"""
    domain = request.query.get("domain", "").strip()
    ip = ddns_service.get_ip(domain)

    if ip:
        return web.json_response({"code": 200, "domain": domain, "ip": ip})
    return web.json_response({"code": 404, "msg": "域名不存在"})

async def handle_list(request):
    """列出所有域名记录（仅调试用）"""
    token = request.query.get("token", "")
    if token != CONFIG["TOKEN"]:
        return web.json_response({"code": 403, "msg": "无权限"})
    return web.json_response({"code": 200, "data": ddns_service.records})

# ==================== 启动服务器 ====================
def main():
    app = web.Application()
    # 注册路由
    app.add_routes([
        web.get("/update", handle_update),   # 上报IP
        web.get("/query", handle_query),     # 查询IP
        web.get("/list", handle_list),       # 查看所有记录
    ])

    # 启动服务
    web.run_app(
        app,
        host=CONFIG["HOST"],
        port=CONFIG["PORT"],
    )

if __name__ == "__main__":
    print(f"DDNS 服务启动: http://{CONFIG['HOST']}:{CONFIG['PORT']}")
    main()