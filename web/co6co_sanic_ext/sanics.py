from __future__ import annotations
from sanic import Sanic, utils, Blueprint
from sanic.blueprint_group import BlueprintGroup
from sanic_routing import Route
from typing import Optional, Callable, Tuple, Dict, List
from pathlib import Path
from co6co.utils import log, File

from sanic.worker.loader import AppLoader
from functools import partial
from co6co.utils.singleton import singleton
from co6co_sanic_ext.view_model import BaseView
from co6co_sanic_ext.api import add_routes
from datetime import datetime
from co6co.utils.source import compile_source
import inspect
"""
工作进程的生命周期

"""


def _create_App(name: str = "__mp_main__", configFile: str = None, apiInit: Callable[[Sanic, Dict, Sanic | None],  None] = None):
    """
    创建应用
    将 config 中的配置信息加载到app.config中
    :param name: 应用名称
    :param config: 配置文件路径[json,py]
    :param apiMount: 挂载api

    return: app
    app.config.web_setting --> 配置信息
    """
    try:
        app = Sanic(name)
        data = locals()
        # primary = data.get("app", None)
        # app.ctx.mainApp = primary
        if configFile == None:
            raise PermissionError("config")
        if app.config != None:
            app.config.update({"web_setting": {'port': 8084, "backlog": 1024, 'host': '0.0.0.0', 'debug': False, 'access_log': True,  'dev': False}})
            customConfig = None
            if '.json' in configFile:
                customConfig = File.File.readJsonFile(configFile)
            else:
                customConfig = utils.load_module_from_file_location(Path(configFile)).configs
            # print(customConfig)
            if customConfig != None:
                app.config.update(customConfig)
            # log.succ(f"app 配置信息：\n{app.config}")
            setting: dict = app.config.web_setting
            app.prepare(**setting)
            if apiInit != None:
                sig = inspect.signature(apiInit)
                params = sig.parameters
                all_param = [app, customConfig, data]
                # all_param = {"app":app,"customConfig": customConfig,"primary": primary}
                # arg_values = {}
                # for param_name, param in params.items():
                #    #如果参数没有默认值
                #    if param.default != inspect.Parameter.empty:
                #        arg_values[param_name] = param.default
                #    else:
                #        arg_values[param_name] = None
                par_len = len(params.items())
                apiInit(*all_param[:par_len])
        else:
            raise PermissionError("app config")
        return app
    except Exception as e:
        log.err(f"创建应用失败：\n{e}{repr(e)}\n 配置信息：{app.config}")
        raise


def startApp(configFile: str, apiInit: Callable[[Sanic, Dict], None]):
    # all_param = {**locals()}
    loader = AppLoader(factory=partial(_create_App, configFile=configFile, apiInit=apiInit))
    # 没有 primary serve 调用loader创建一个个
    Sanic.serve(app_loader=loader)


@singleton
class ViewManage:
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
    bluePrint: Blueprint = None
    createTime: datetime = None

    @staticmethod
    def static_fun():
        """
        当静态方法遇上,单例模式中的
        """
        print("ddd")

    def __init__(self,  app: Sanic) -> None:
        super().__init__()
        self.viewDict = {}
        self.app = App(app)

    def _createBlue(self, blueName: str, url_prefix: str, version: int | str | float | None = 1, *views: BaseView):
        blue = Blueprint(blueName, url_prefix=url_prefix, version=version)
        add_routes(blue, *views)
        return blue

    def exist(self, blueName):
        return blueName in self.app.app.blueprints

    def add(self, blueName: str, url_prefix: str, version: int | str | float | None = 1, *views: BaseView):
        """
        BluePrint 名字不能与系统中存在的名字重复
        请求URL: /v{version}}/${url_prefix}/{BaseView.routePath}
        """
        blue = self._createBlue(blueName, url_prefix, version, *views)
        self.app.app.blueprint(blue)

    def _getUrls(self, url_prefix: str, version: int | str | float | None = 1, *views: BaseView):
        urls = ("/v{}{}{}".format(version, url_prefix, v.routePath) for v in views)
        return urls

    def replace(self, blueName: str, url_prefix: str, version: int | str | float | None = 1, *views: BaseView):
        if self.exist(blueName):
            blue = self._createBlue(blueName, url_prefix, version, *views)
            urls = self._getUrls(url_prefix, version, *views)
            self.app.remove_route(*urls)
            # 方法1 移除在按增加的来 服务器停止
            self.app.app.blueprints.pop(blueName)
            self.app.app.router.reset()
            self.app.app.blueprint(blue)
            # 方法2  简单替换无法实现功能，替换完还是以前的功能
            # self.app.app.blueprints[blueName]=blue
        else:
            raise Exception("Blueprint {} is Null".format(blueName))


class App:
    app: Sanic = None

    def __init__(self, app: Sanic = None) -> None:
        if app == None:
            app = Sanic.get_app()
        self.app = app
        pass

    def findRouteWithStart(self, url_prefix: str):
        routes = [r for r in self.app.router.routes if r.uri.startswith(url_prefix)]
        return routes

    def findOneRoute(self, uri: str):
        routes = [r for r in self.app.router.routes if r.uri == uri]
        if len(routes) == 1:
            return routes[0]
        return None

    def remove_route(self, *uri: str):
        """
        动态删除路由
        """
        routes = [r for r in self.app.router.routes if r.uri in uri]

        for r in routes:
            if r in self.app.router.routes:
                # self.app.router.reset()
                del self.app.router.routes[r]  # 元组无发删除

    # 动态替换路由
    def replace_route(self, uri, handler, methods=None):
        """
        替换路由
        """
        self.remove_route(uri)
        self.app.add_route(handler, uri, methods=methods)

    @staticmethod
    def appendView(app: Sanic, *viewSource: str,  blueName: str = "user_append_View", url_prefix="api", version=1, ingoreView: List[str] = ["AuthMethodView", 'BaseMethodView']):
        """
        增加视图
        前置条件: 1. 视图名不能重名
                 2. api地址不能重复
                 3.
        """
        try:
            viewMage = ViewManage(app)
            AllView: List[BaseView] = []
            nameList = []  # 路由名称不能重复
            routeUrl = []

            if len(viewSource) > 0:
                for s in viewSource:
                    globals_vars = {}
                    compile_source(s, globals_vars)
                    views: List[BaseView] = [globals_vars[i] for i in globals_vars if str(i).endswith("View") and i not in ingoreView]
                    for v in views:
                        if v.__name__ in nameList:
                            log.warn("视图名称‘{}’重复".format(v.__name__))
                            continue
                        if v.routePath in routeUrl:
                            log.warn("视图路由‘{}’重复".format(v.routePath))
                            continue
                        nameList.append(v.__name__)
                        routeUrl.append(v.routePath)
                        AllView.append(v)
            if len(AllView) > 0:
                viewMage.add(blueName, url_prefix, version, *AllView)
        except Exception as e:
            log.err("动态模块失败", e)
