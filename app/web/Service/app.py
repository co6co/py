from __future__ import annotations
from sanic import Sanic
from co6co.utils import log
from sanic.config import Config
from co6co_sanic_ext.utils.cors_utils import attach_cors
from co6co_sanic_ext import sanics
from co6co_sanic_ext import session
from co6co_web_db.services.db_service import injectDbSessionFactory
from cacheout import Cache
import time
from co6co_task.service.RouterService import dynamicRouter
from co6co_task.service.Tasks import TasksMgr


def appendRoute(app: Sanic):
    try:
        bll = dynamicRouter(app.config.db_settings)
        source = bll.getSourceList()
        sanics .App.appendView(app, *source, blueName="user_append_View")
        pass
    except Exception as e:
        log.err("动态模块失败", e)


def init(app: Sanic, _: dict):
    """
    初始化
    """
    # log.warn("APP:",   id(app))
    attach_cors(app)
    from api import api
    injectDbSessionFactory(app, app.config.db_settings)
    '''
    @app.main_process_start
    async def setup(app: Sanic, _: Config):
        pass
        # mainApp = app.ctx.mainApp  # 主进程
        # task = TasksMgr(app)
        # task.startTimeTask()
        # log.warn("setup:", type(_), id(app), app.ctx) 
    @app.main_process_stop
    async def _setup(app: Sanic, _: Config):
        if hasattr(app.ctx, "taskMgr"):
            task: TasksMgr = app.ctx.taskMgr
            task.stop()
        pass'
    '''
    @app.before_server_start
    async def _(app: Sanic, _: Config):
        cache = Cache(maxsize=256, ttl=30, timer=time.time, default=None)
        app.ctx.Cache = cache
        pass
    app.blueprint(api)
    dynamicRouter.appendRoute(app)
    session.init(app)


def main(config: dict):
    sanics.startApp(config, init, TasksMgr.create_instance)
    log.succ("***,main", __file__) # 子进程不执行



if __name__ == "__main__":
    # print("__file__", __file__)
    # current_file_path = os.path.abspath(__file__)
    # print("basename", os.path.basename(__file__))
    # print("dirname", os.path.dirname(__file__))
    # print("__file__ dir:", current_file_path)
    # current_directory = os.getcwd()
    # print("当前工作目录：", current_directory)
    # print("当前dir：", os.path.curdir)
    log.succ("##", __name__)
    config = sanics.getConfig(sanics.getConfigFilder(__file__))
    main(config)
