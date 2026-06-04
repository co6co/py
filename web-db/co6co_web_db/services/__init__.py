from sanic import Sanic, Request 
from sanic.log import logger 
from co6co_db_ext.db_session import db_service 
import multiprocessing
from multiprocessing.managers import DictProxy
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import scoped_session

def get_db_service(app: Sanic) -> db_service:
    return app.ctx.service
def get_cache(app: Sanic) -> DictProxy:
    return app.shared_ctx.cache 
def get_db_session(request: Request) -> AsyncSession | scoped_session:
    """
    获取db session
    """
    session = request.ctx.session
    if isinstance(session, AsyncSession):
        return session
    elif isinstance(session, scoped_session):
        return session
    raise Exception("未实现DbSession")

def injectDbSessionFactory(app: Sanic, settings: dict = {}, engineUrl: str = None, sessionApi: list = ["/api"],init_tables: bool = True):
    """
    挂在 DBSession_factory
    """
    service: db_service = None
    # sanic_session 包使用了 cockie 未能满足使用需求 已修改其他方式的 session 实现
    # session_interface = InMemorySessionInterface(session_name="mem_session", cookie_name=app.name, prefix=app.name)
    # Session(app, interface=session_interface)
    #  settings 与 settings is not None 结果不同
   
    if settings or engineUrl is not None: 
        service = db_service(settings, engineUrl)
        app.ctx.service = service
        if init_tables:
            service.sync_init_tables() 

    @app.main_process_start
    async def inject_session_factory(app: Sanic): 
        app.shared_ctx.cache = multiprocessing.Manager().dict()
        # app.shared_ctx.cache["db_session_factory"]=_async_session_factory

    def checkApi(request: Request):
        for api in sessionApi:
            if api in request.path:
                return True
        return False

    @app.middleware("request")
    async def inject_session(request: Request):
        # await session_interface.open(request)
        if checkApi(request):
            # logger.info("mount DbSession 。。。")
            if service is not None and service.useAsync:
                request.app.ctx.session_fatcory = service.async_session_factory
                request.ctx.session = service.async_session_factory()
                request.ctx.session_ctx_token = service.base_model_session_ctx.set(request.ctx.session)
            elif service is not None:
                request.ctx.session = service.session

    @app.middleware("response")
    async def close_session(request: Request, response):
        # await session_interface.save(request, response)
        if hasattr(request.ctx, "session_ctx_token"):
            try:
                if service is not None and service.useAsync:
                    service.base_model_session_ctx.reset(request.ctx.session_ctx_token)
                    await request.ctx.session.close()
                # logger.info("close DbSession。")
                # await request.ctx.session.dispose()
            except Exception as e:
                logger.err(e)
                # logger.error("close DbSession。Error")
