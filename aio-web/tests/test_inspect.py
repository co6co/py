import pytest
import inspect
from aiohttp import web, http_parser, web_protocol, web_server
import asyncio
from yarl import URL

async def c():
    return 1 
def b():
    return 2 
class d:
    def __init__(self):
        pass
 
class demoView:
    """
    await demoView(...)
    ==>
        view_instance = demoView2(request, session)  # 创建实例
        response = await view_instance()  # 调用 __call__ 方法
    """
    def __init__(self, request: web.Request, *args):
        self.request = request
        pass
    async def get(self, request: web.Request):
        return web.json_response({"message": "Hello, World!"}) 
    async def __call__(self) -> web.StreamResponse:
        method = self.request.method
        handler = getattr(self, method.lower(), None)
        if handler is None:
            raise web.HTTPMethodNotAllowed(method, [])
        return await handler(self.request)
    def __await__(self)  :
        return self.__call__().__await__()


class demoView2(web.View):
    def __init__(self, request: web.Request, *args):
        super().__init__(request)
        pass 
    async def get(self):
        return web.json_response({"message": "Hello, World!"}) 

def test_corator():
    assert inspect.iscoroutinefunction(c)
    assert not inspect.iscoroutinefunction(b)
    assert not inspect.iscoroutinefunction(demoView2)

def create_request(loop): 
    #loop = asyncio .new_event_loop()
    message = http_parser.RawRequestMessage(
            "GET", "/", "HTTP/1.1", "", "", True, None, True, True, URL("/")
    )
    protocol = web_protocol.RequestHandler(
        web_server.Server(lambda x: None), loop=loop
    )
    ploadload = http_parser.StreamReader(protocol, limit=1024)
    request = web.Request(
        message, ploadload, protocol, None, None, loop
    )
     
    return request

def test_custom_view( ):
    handler = demoView
    async def demo_custom(): 
        assert not issubclass(handler, web.View)
        assert isinstance(handler, type) 
        request = create_request(asyncio.get_event_loop())
        print(dir(handler),hasattr(handler, "__await__"))
        if hasattr(handler, "__await__"):
            ## 两步合一步
            print("两步合一步")
            result= await handler(request, 1)
        else: 
            instance= handler(request, 1)
            result=await instance()   # 

        assert not isinstance(result, handler)  #
        print(result,"不是handler 的实例，是",type(result)) 

    asyncio.run(demo_custom())

def test_view( ):
    handler = demoView2
    assert issubclass(handler, web.View)
    assert isinstance(handler, type) 
    async def demo_test(): 
        assert issubclass(handler, web.View)
        assert isinstance(handler, type) 
        request = create_request(asyncio.get_event_loop()) 
        result= await handler(request, 1)
        assert not isinstance(result, handler)  #
        print(result,"不是handler 的实例，是",type(result)) 

    asyncio.run(demo_test())
    pass
