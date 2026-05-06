from aiohttp import web
import ssl,asyncio,os 
import abc
class aio_http_server_base(metaclass=abc.ABCMeta):
    def __init__(self,port:int) -> None: 
        self._ssl=False
        self._port=port
        self.cert = None
        self.key = None
    def use_ssl(self, cert_path:str, key_path:str):
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
        ssl_context.load_cert_chain(self.cert, self.key )
        return ssl_context
    async def start(self,host="0.0.0.0"): 
        app = web.Application() 
        await self.route_handler(app) 
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, self.port,ssl_context=self.create_ssl_context())
        await site.start() 
        # 保持服务器运行
        await asyncio.Future()
        

    @abc.abstractmethod
    async def route_handler(self,app: web.Application):
        """下面为实例代码 请在子类中实现该方法"""
        app.add_routes([
            #web.get('/ws', self.websocket_handler),
            #web.get('/status', self.status_handler),
            #web.get('/api/rooms', self.rooms_handler),
            web.static('/', os.path.join(os.path.dirname(__file__), 'pages')),
        ])
        raise NotImplementedError