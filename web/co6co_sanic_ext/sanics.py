from __future__ import annotations
from sanic import Sanic, utils, Blueprint
from sanic.blueprint_group import BlueprintGroup
from typing import Optional, Callable, Any, Dict, List
from pathlib import Path
from co6co.utils import log, File

from sanic.worker.loader import AppLoader
from functools import partial
from co6co.utils.singleton import Singleton
from co6co_sanic_ext.view_model import BaseView
from co6co_sanic_ext.api import add_routes


def _create_App(name: str = "__mp_main__", config: str = None, apiMount: Optional[Callable[[Sanic, Dict],  None]] = None):
    try:
        app = Sanic(name)
        if config == None:
            raise PermissionError("config")
        if app.config != None:
            app.config.update({"web_setting": {'port': 8084, 'host': '0.0.0.0', 'debug': False, 'access_log': True,  'dev': False}})
            customConfig = None
            if '.json' in config:
                customConfig = File.File.readJsonFile(config)
            else:
                customConfig = utils.load_module_from_file_location(Path(config)).configs
            if customConfig != None:
                app.config.update(customConfig)
            # log.succ(f"app 配置信息：\n{app.config}")
            if apiMount != None:
                apiMount(app, customConfig)

        return app
    except Exception as e:
        log.err(f"创建应用失败：\n{e}{repr(e)}\n 配置信息：{app.config}")
        raise


def startApp(configFile: str, apiInit: Optional[Callable[[Sanic, Any], None]]):
    loader = AppLoader(factory=partial(_create_App, config=configFile, apiMount=apiInit))
    app = loader.load()
    if app != None and app.config != None:
        setting = app.config.web_setting
        backlog = 1024
        if "backlog" in setting:
            backlog = setting.get("backlog")
        app.prepare(host=setting.get("host"), backlog=backlog, port=setting.get("port"), debug=setting.get("debug"), access_log=setting.get("access_log"), dev=setting.get("dev"))
        Sanic.serve(primary=app, app_loader=loader)
        # app.run(host=setting.get("host"), port=setting.get("port"),debug=True, access_log=True)
    return app


class ViewManage(Singleton):
    """
    目标： 动态增加HTTPMethodView动态 api
    遇到的问题：取消以前增加的蓝图
    处理步骤：
        1. 应用初始化时从数据库中读出所有带增加的功能
        2. 将所有功能放在一个蓝图中， 统一一起增加
        3. 在平台中修改某个功能时需要，删除改功能并重新挂在到蓝图中
        4. 在平台中增加某想功能，需要在蓝图中增加


    """
    viewDict: Dict[str, BaseView] = None
    app:  App = None

    def __init__(self,  app: Sanic) -> None:
        super().__init__()
        self.viewDict = {}
        self.app = App(app)

    def add(self, url_prefix: str, version: int | str | float | None = 1, *views: BaseView):
        api = Blueprint("sysTask_API", url_prefix=url_prefix, version=version)
        add_routes(api, *views)
        self.app.app.blueprint(api)


class App:
    app: Sanic = None

    def __init__(self, app: Sanic = None) -> None:
        if app == None:
            app = Sanic.get_app()
        self.app = app
        pass

    def remove_route(self, uri):
        """
        动态删除路由
        """
        if uri in self.app.router.routes:
            del self.app.router.routes[uri]

    # 动态替换路由

    def replace_route(self, uri, handler, methods=None):
        """
        替换路由
        """
        self.remove_route(uri)
        self.app.add_route(handler, uri, methods=methods)
