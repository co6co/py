from aiohttp import web
import ssl
import asyncio
import os
from aiohttp.typedefs import Middleware
from typing import Iterable
import abc
from co6co.utils import log
from co6co.utils.json_util import JSONEncoder
from co6co.data import DictNamespace
from typing import List

class http_server_base(metaclass=abc.ABCMeta):
    def __init__(self, *,middlewares:List[Middleware]=[],client_max_size:int=1024*1024*100,**kvargs) -> None:
        self._ssl = False 
        self.cert = None
        self.key = None 
        self._app = web.Application(middlewares=middlewares,**kvargs)
    @property
    def app(self):
        return self._app
    def response_json(self, data, status=200):
        return web.json_response(data, content_type='application/json', status=status, dumps=JSONEncoder.dumps)

    def use_ssl(self, cert_path: str, key_path: str):
        self.cert = cert_path
        self.key = key_path
        self._ssl = True

    @property
    def ssl(self):
        return self._ssl

    @property
    def port(self):
        return self._port

    def create_ssl_context(self):
        # 配置 SSL 上下文
        if not self._ssl:
            return None
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(self.cert, self.key)
        return ssl_context
 
    async def before_start(self):
        """
        服务器启动前的初始化操作
        可以在子类中实现，例如加载配置、初始化数据库等
        return:  web.TCPSite 参数
        """
        return {} 
    async def _start_site_start(self,runner : web.AppRunner,host:str,port:int,**kvargs):
        """
        开始启动site
        """
        site = web.TCPSite(runner, host, port, ssl_context=self.create_ssl_context(),**kvargs) 
        await site.start() 
    async def start(self, host="0.0.0.0",port:int=8080 ):
        """
        开始服务 
        """
        siteconfig = await self.before_start() 
        await self.route_handler(self.app)
        runner = web.AppRunner(self.app, handle_signals=True) 
        try:
            await runner.setup() 
            # site0 = web.TCPSite(runner, '127.0.0.1', 8080), 
            await self._start_site_start(runner, host,port, **siteconfig) 
            await self._print_info(runner )
            await asyncio.Future()  # 等待直到应用关闭（通过信号）
        except asyncio.CancelledError as e:
            print(f"signal received ... {e}")
        finally:
            # 清理
            print("stopping server ...") 
            await runner.cleanup() 
            print("server stopped.")
            # 保持服务器运行

    async def get_route_info(self):
        """获取服务器路由信息"""
        routes_info = []
        for route in self.app.router.routes():
            resource = route.resource
            if isinstance(resource, web.PlainResource):
                routes_info.append(DictNamespace(method=route.method, path=resource.canonical))
            # elif isinstance(resource, web.StaticResource):
            #    routes_info.append(DictNamespace(method=route.method, path=resource.get_info()))
        return routes_info

    async def _print_info(self,  runner: web.AppRunner):
        """打印服务器信息"""
        for s in runner._sites:
            # s: web.TCPSite = s
            # w, h = ("ws", "http")
            # if self.ssl:
            #    w, h = ("wss", "https")
            log.info(f"server running on: {s.name}")
            # log.info(f"server running on: {h}://{s._host}:{s._port}/")
            # 获取所有路由
            routes_info = await self.get_route_info()
            for r in routes_info:
                log.info(f"Method:\t{r.method}\tPath:\t{ r.path} ")

    @abc.abstractmethod
    async def route_handler(self, app: web.Application):
        """下面为实例代码 请在子类中实现该方法"""
        import pathlib
        app.add_routes([
            # web.get('/ws', self.websocket_handler),
            # web.get('/status', self.status_handler),
            # web.get('/api/rooms', self.rooms_handler),
            # web.static('/', os.path.join(os.path.dirname(__file__), 'pages')),
            web.static('/', pathlib.Path.cwd() / 'pages'),
            # web.view("api/demo", demoView),
        ])
        # app.router.add_route("*", "/demo2", demoView),
        raise NotImplementedError
