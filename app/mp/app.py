from sanic import Sanic,utils
from co6co.utils import log ,File
from sanic import utils
from co6co_sanic_ext.utils.cors_utils import attach_cors
from co6co_sanic_ext import startApp

from sanic.request import Request
from sanic.response import text,json 
import argparse

from pathlib import Path  

def init (app:Sanic,customConfig):
    log.warn(customConfig)
    attach_cors(app) 
    from api import api
    #from static import res 
    #app.blueprint(res) 
    #injectDbSessionFactory(app,app.config.db_settings) 
    app.blueprint(api) 
    

if __name__ == "__main__": 
    parser=argparse.ArgumentParser(description="audit service.")
    parser.add_argument("-c","--config",default="app_config.json")
    args=parser.parse_args() 
    startApp(args.config,init) 


