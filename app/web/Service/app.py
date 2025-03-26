from __future__ import annotations

from sanic import Sanic
from co6co.utils import log
from sanic.config import Config
from co6co_sanic_ext.utils.cors_utils import attach_cors
from co6co_sanic_ext import sanics
from co6co_sanic_ext import session
from co6co_web_db.services.db_service import injectDbSessionFactory
import argparse
from cacheout import Cache
import time
import os
from services.taskService import dynamicRouter
from services.tasks.main import TasksMgr


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
    attach_cors(app)
    from api import api
    injectDbSessionFactory(app, app.config.db_settings)

    @app.main_process_start
    async def _(app: Sanic, _: Config):
        task = TasksMgr(app)
        for k in app._app_registry:
            log.warn("main_process_start", k, app._app_registry[k], id(app._app_registry[k]))
            app._app_registry[k].ctx.taskMgr = task
            log.warn("setting", app._app_registry[k].ctx.taskMgr)

        log.succ("main_process_start", id(app._state.app), id(id), dir(app._state))
        app.ctx.taskMgr = task
        # task.startTimeTask()
        pass

    @app.before_server_start
    async def _(app: Sanic, _: Config):
        cache = Cache(maxsize=256, ttl=30, timer=time.time, default=None)
        app.ctx.Cache = cache

        # app.ctx.taskMgr = mainApp.ctx.taskMgr
        # log.warn("before_server_start:", type(_),  mainApp)

        log.succ("before_server_start:", type(_), id(app), app.ctx)
        log.warn("read", app.ctx.taskMgr)
        pass

    @app.main_process_stop
    async def _(app: Sanic, _: Config):
        task: TasksMgr = app.ctx.taskMgr
        task.stop()
        pass
    app.blueprint(api)
    appendRoute(app)
    session.init(app)


def main():
    dir = os.path.dirname(__file__)
    defaultConfig = "{}/app_config.json".format(dir)
    configPath = os.path.abspath(defaultConfig)
    parser = argparse.ArgumentParser(description="System Service.")
    parser.add_argument("-c", "--config", default=configPath, help="default:{}".format(configPath))
    args = parser.parse_args()
    sanics.startApp(args.config, init)


if __name__ == "__main__":
    # print("__file__", __file__)
    # current_file_path = os.path.abspath(__file__)
    # print("basename", os.path.basename(__file__))
    # print("dirname", os.path.dirname(__file__))
    # print("__file__ dir:", current_file_path)
    # current_directory = os.getcwd()
    # print("当前工作目录：", current_directory)
    # print("当前dir：", os.path.curdir)
    main()
