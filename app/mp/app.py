from __future__ import annotations  

from sanic import Sanic,utils
from co6co.utils import log ,File
from sanic import utils
from co6co_sanic_ext.utils.cors_utils import attach_cors
from co6co_sanic_ext import sanics
from co6co_web_db.services.db_service import injectDbSessionFactory 
from co6co_db_ext.db_session import db_service

from sanic.request import Request
from sanic.response import text,json 
import argparse 
from co6co_sanic_ext .model.res .result import Result
from pathlib import Path
from cacheout import Cache
import time


from view_model.tasks.devices_tasks import update_device_poster_task

'''from model.pos.right import *  
from model.pos.biz import *
from model.pos.wx import * '''

from model.pos.biz import BasePO, bizResourcePO  ,bizAlarmAttachPO,bizAlarmTypePO ,bizAlarmPO

def init (app:Sanic,customConfig):
    """
    公众号12345
    初始化
    """

    #@app.main_process_start 
    #async def on_main_process_start(app: Sanic):
    #    update_device_poster_task(app)

    log.warn(customConfig)
    attach_cors(app) 
    from api import api
    from api.hwx import hwx_api
    #from static import res 
    #app.blueprint(res) 
    #service=db_service(app,app.config.db_settings,BasePO)
    #service.sync_init_tables()  
    injectDbSessionFactory(app,app.config.db_settings,sessionApi=["/api","/nyzh"]) 
    app.blueprint(api)
    app.blueprint(hwx_api)
    
    app.add_task(update_device_poster_task(app))  
    cache = Cache(maxsize=256, ttl=30, timer=time.time, default=None) 
    app.ctx.Cache=cache 

    @app.main_process_start
    async def main_process_start(*_):
        print(">>>>>>main_process_start_start <<<<<<")

    @app.before_server_start
    async def server_start(*_):
        print(">>>>>>before_server_start <<<<<<")
    @app.after_server_start
    async def server_started(*_):
        print(">>>>>>after_server_start <<<<<<")

    @app.before_server_stop
    async def app_before_stop(*_):
        print(">>>>>>before_server_stop <<<<<<")
    @app.after_server_stop
    async def after_server_stop(*_):
        print(">>>>>>after_server_stop <<<<<<")

    @app.main_process_stop
    async def main_process_stop(*_):
        print(">>>>>>main_process_stop <<<<<<")

    '''
    @app.middleware("request")
    async def inject_session(request:Request):  
        #logger.info("mount DbSession 。。。")
        if "/nyzh/pubApi/ipncAlive" in request.path:
            print("原始数据：",request.headers,request.body)
    ''' 
if __name__ == "__main__":     
    parser=argparse.ArgumentParser(description="audit service.")
    parser.add_argument("-c","--config",default="app_config.json")
    args=parser.parse_args() 
    sanics.startApp(args.config,init) 
    


