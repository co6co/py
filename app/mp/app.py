from sanic import Sanic,utils
from co6co.utils import log ,File
from sanic import utils
from co6co_sanic_ext.utils.cors_utils import attach_cors
from co6co_sanic_ext import sanics
from co6co_web_db.services.db_service import injectDbSessionFactory,db_service

from sanic.request import Request
from sanic.response import text,json 
import argparse
from co6co_db_ext.po import BasePO

from pathlib import Path  

def init (app:Sanic,customConfig):
    """
    公众号12345
    初始化
    """
    log.warn(customConfig)
    attach_cors(app) 
    from api import api
    #from static import res 
    #app.blueprint(res) 
    #service=db_service(app,app.config.db_settings,BasePO)
   # service.sync_init_tables() 
    injectDbSessionFactory(app,app.config.db_settings,BasePO) 
    app.blueprint(api)  
    

if __name__ == "__main__": 
    parser=argparse.ArgumentParser(description="audit service.")
    parser.add_argument("-c","--config",default="app_config.json")
    args=parser.parse_args() 
    sanics.startApp(args.config,init) 


