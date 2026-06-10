from sanic import Sanic, Request
from sanic.response import HTTPResponse
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

def set_rollback(request: Request, rollback: bool):
    """
    设置回滚事务
    """
    request.ctx.rollback = rollback

def get_context_session(request: Request)->AsyncSession|scoped_session|None:
    service=get_db_service(request.app)
    if hasattr(request.ctx, "session"): 
        return service.context_var.get() 
    return None


def injectDbSessionFactory(app: Sanic, settings: dict = {}, engineUrl: str = None, sessionApi: list = ["/api"],init_tables: bool = True):
    """
    挂在 DBSession_factory
    :param app: Sanic 应用实例
    :param settings: 数据库配置
    :param engineUrl: 数据库连接字符串
    :param sessionApi: 是否使用数据库Session
    :param init_tables: 是否初始化数据库表
    :return: None
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
                _session=service.Session()
                request.ctx.session = _session 
                # 开启事务（AsyncSession 自动开启，也可显式开启）
                request.ctx.rollback = False  # 主动标记不回滚事务
                await _session.begin()
            elif service is not None:
                request.ctx.session = service.session
            request.ctx.session_ctx_token = service.context_var.set(request.ctx.session)
    async def commit_or_rollback(request: Request, response:HTTPResponse):
        """
        提交或回滚事务
        """
        if not hasattr(request.ctx, "session"):
            return
        try:
            session: AsyncSession   = request.ctx.session
            if getattr(request.ctx, "rollback", False):
                await session.rollback()
            elif 200 <= response.status < 300:
                await session.commit()
            else:
                await session.rollback()
        except Exception as e:
            logger.error("commit_or_rollback,error",exc_info=e)
            await session.rollback()
        finally:
            await session.close()
    @app.middleware("response")
    async def close_session(request: Request, response:HTTPResponse):
        # await session_interface.save(request, response)
        if hasattr(request.ctx, "session"):
            try:
                service.context_var.reset(request.ctx.session_ctx_token)
                if service is not None and service.useAsync:
                 
                    await commit_or_rollback(request, response)
                    await request.ctx.session.close()
                # logger.info("close DbSession。")
                # await request.ctx.session.dispose()
            except Exception as e:
                logger.error("close_session",e)
                # logger.error("close DbSession。Error")
