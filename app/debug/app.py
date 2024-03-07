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
from tasks.test1 import monitor

  
def init (app:Sanic,customConfig):
    """ 
    初始化
    """ 
    log.warn(customConfig)
    attach_cors(app) 
    from api import api
     
    injectDbSessionFactory(app,None,sessionApi=["/api"]) 
    app.blueprint(api) 
    app.add_task(monitor)
    
if __name__ == "__main__":     
    parser=argparse.ArgumentParser(description="audit service.")
    parser.add_argument("-c","--config",default="app_config.json")
    args=parser.parse_args() 
    sanics.startApp(args.config,init) 
    


