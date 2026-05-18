import inspect
from aiohttp import web
from .viewbase import ViewBase
from co6co_db_ext.db_session import db_service
from co6co_db_ext .actuator import Actuator
from co6co_db_ext.jwt_service import JwtService  
from co6co.utils import log
from co6co.data.result import Result

async def jwt_middleware(request: web.Request, handler):
    """JWT中间件"""
    config = request.app.config
    authorization=request.headers.get("Authorization") 
    jwt= JwtService(config.get("web_settings").get("jwt_secret"))
    userData=jwt.decode(authorization)
    request.userData=userData
    return await handler(request)



@web.middleware
async def db_middleware(request: web.Request, handler):
    """数据库会话中间件"""
    db: db_service = request.app.db
    paramerName = "actuator"
    authonParam = "userData"
    config = request.app.config
    # 获取handler的签名
    try: 
        kwargs = {}
        sig = inspect.signature(handler)
        isCoroutine = inspect.iscoroutinefunction(handler)
        isView = isinstance(handler, type) and issubclass(handler, web.View)
        #print("是视图模型", isView)
        isWaitable = isCoroutine or isView
        useSession = False
        response = None 
        # 如果 handler 需要 db 参数，则注入
        for name, param in sig.parameters.items():
            #log.warn(name,sig.parameters,"request" in sig.parameters ,type(handler))
            #log.warn("name->", name,"param->", param) # name->  request param-> request: aiohttp.web_request.Request
            if name == paramerName:
                useSession = True
            elif name == "request":
                kwargs[name] = request
            elif name == "userData":
                try: 
                    authorization=request.headers.get("Authorization") 
                    jwt= JwtService(config.get("web_settings").get("jwt_secret"))
                    userData=jwt.decode(authorization)
                    kwargs[name] = userData
                except Exception as e:
                    log.err(f"请求认证过程中{request.method}{request.path}发生异常: {e}", e)
                    response =  ViewBase.response_json(Result.fail(str(e)),status=401)
                    return response
        if useSession: 
            #async with session_context(db.Session())() as session: 
            async with db.Session() as session, session.begin(): 
                kwargs.update({paramerName:  Actuator(session) })
                # 调用 handler 并传入参数
                try:
                    if isWaitable:
                        response = await handler(**kwargs)
                    else:
                        response = handler(**kwargs)
                    #await session.commit()
                except Exception as e:
                    await session.rollback()
                    log.err(
                        f"请求过程中{request.method}{request.path}发生异常: {e}", e
                    )
                    response =  ViewBase.response_json(Result.fail(str(e)))
        else:
            if isWaitable:
                response = await handler(request)
            else:
                response = handler(request)
        return response

    except ValueError:
        # 如果handler不是可调用的，或者无法获取签名
        print(f"Handler: {handler} is not callable or has no signature")
    except Exception as e:
        log.err(f"请求过程中{request.method}{request.path}发生异常: {e}", e)