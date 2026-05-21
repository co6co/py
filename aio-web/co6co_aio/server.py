
from aiohttp import web
import ssl
import asyncio
import os
from aiohttp.typedefs import Middleware
from typing import Iterable,TypeVar,cast,Optional
import abc
from co6co.utils import log
from co6co.utils.json_util import JSONEncoder
from co6co.data import DictNamespace
from typing import List
from co6co_db_ext.db_session import db_service,connectSetting
import json
from dataclasses import dataclass 
from co6co_db_ext.jwt_service import JwtService

T = TypeVar("T",bound='http_server_base')

@dataclass(init=False)
class webConfig:
    port: int
    host: str
    ssl: bool
    cert: str
    key: str
    jwt_secret: str

    def post_init(self, config: DictNamespace):
        self.port = config.port
        self.host = config.host
        self.ssl = config.ssl
        self.cert = config.cert
        self.key = config.key
        self.jwt_secret = config.jwt_secret
        

@dataclass
class AppConfig:
    raw: dict
    web: Optional[webConfig] = None
    db: Optional[connectSetting] = None

    @staticmethod
    def get_config(config_file: str, *,use_web_config:bool=True,use_db_config:bool=True):
        appConfig=AppConfig({})
        try:
            with open(config_file, "r", encoding="utf-8") as f: 
                appConfig.raw =json.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"配置文件不存在: {config_file}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"配置文件 JSON 解析失败: {e}")  
        if use_web_config:
            appConfig.web = AppConfig.get_web_config(appConfig.raw)
        if use_db_config: 
            appConfig.db = AppConfig.get_db_config(appConfig.raw)
        return appConfig
    
    @staticmethod
    def get_web_config(configs:dict):
        try: 
            setting = configs.get("web_setting", {})
            web_config = webConfig()
            web_config.post_init(DictNamespace(**setting)) 
            return web_config
        except Exception as e:
            log.err(f"web_setting error:{setting}",e)
            raise e
    @staticmethod
    def get_db_config(configs:dict):
        try:
            setting = configs.get("db_settings", {})
            data=DictNamespace(**setting) 
            _config = connectSetting.create_default(data) 
            return _config
        except Exception as e:
            log.err(f"db_settings error:{setting}",e)
            raise e

async def _appStart(app: web.Application):
    """应用启动时的初始化操作"""
    config  =  cast(AppConfig, app.config) 
    db = db_service(config.db)
    app.db = db
    jwt= JwtService(config.web.jwt_secret)
    app.jwtService = jwt 
    log.warn("所有配置",config)
    pass

def init_db(configFilePath:str):
    _,_,db_config=http_server_base.get_config(configFilePath)
    db = db_service(db_config)
    db.sync_init_tables() 

async def _appShutdown(app: web.Application):
    """应用关闭时的清理操作"""
    print("应用关闭时的清理操作", id(app))
    pass 
class http_server_base(metaclass=abc.ABCMeta):
    def __init__(
        self, 
        *,
        middlewares: List[Middleware] = [],
        client_max_size: int = 1024 * 1024 * 100,
        **kvargs,
    ) -> None:
        self._ssl = False
        self.cert = None
        self.key = None
        self._app = web.Application(middlewares=middlewares, **kvargs) 

    @classmethod
    def _create_server(cls ,configFile:str,*, middlewares: List[Middleware] = [],
        client_max_size: int = 1024 * 1024 * 100,
        **kvargs):
        appConfig= AppConfig.get_config(configFile) 
        instance= cls(middlewares=middlewares,client_max_size=client_max_size,**kvargs) 
        instance.app.config = appConfig  
        if appConfig.web.ssl:
            instance.use_ssl( appConfig.web.cert,  appConfig.web.key)
        instance.app.on_startup.append(_appStart)
        instance.app.on_shutdown.append(_appShutdown)
        return instance,appConfig
    @classmethod
    def create_server(cls ,configFile:str,*, middlewares: List[Middleware] = [],
        client_max_size: int = 1024 * 1024 * 100,
        **kvargs):
        instance,_ = cls._create_server(configFile, middlewares=middlewares, client_max_size=client_max_size, **kvargs)
        return instance

    @classmethod
    def start_server(cls ,configFile:str,*, middlewares: List[Middleware] = [],client_max_size: int = 1024 * 1024 * 100,
        **kvargs):
        instance,appConfig  = cls._create_server(configFile, middlewares=middlewares, client_max_size=client_max_size, **kvargs)
        asyncio.run(instance.start(appConfig.web.host,appConfig.web.port)) 

    @property
    def app(self):
        return self._app

    def response_json(self, data, status=200):
        return web.json_response(
            data,
            content_type="application/json",
            status=status,
            dumps=JSONEncoder.dumps,
        )

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

    async def _start_site_start(
        self, runner: web.AppRunner, host: str, port: int, **kvargs
    ):
        """
        开始启动site
        """
        site = web.TCPSite(
            runner, host, port, ssl_context=self.create_ssl_context(), **kvargs
        )
        await site.start()

    async def start(self, host="0.0.0.0", port: int = 8080):
        """
        开始服务
        """
        siteconfig = await self.before_start()
        await self.route_handler(self.app)
        runner = web.AppRunner(self.app, handle_signals=True)
        try:
            await runner.setup()
            # site0 = web.TCPSite(runner, '127.0.0.1', 8080),
            await self._start_site_start(runner, host, port, **siteconfig)
            await self._print_info(runner)
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
                routes_info.append(
                    DictNamespace(method=route.method, path=resource.canonical)
                )
            # elif isinstance(resource, web.StaticResource):
            #    routes_info.append(DictNamespace(method=route.method, path=resource.get_info()))
        return routes_info

    async def _print_info(self, runner: web.AppRunner):
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
                log.info(f"Method:\t{r.method}\tPath:\t{r.path} ")

    @abc.abstractmethod
    async def route_handler(self, app: web.Application):
        """下面为实例代码 请在子类中实现该方法"""
        import pathlib

        app.add_routes(
            [
                # web.get('/ws', self.websocket_handler),
                # web.get('/status', self.status_handler),
                # web.get('/api/rooms', self.rooms_handler),
                # web.static('/', os.path.join(os.path.dirname(__file__), 'pages')),
                web.static("/", pathlib.Path.cwd() / "pages"),
                # web.view("api/demo", demoView),
            ]
        )
        # app.router.add_route("*", "/demo2", demoView),
        raise NotImplementedError
