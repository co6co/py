
from sanic_ext import Extend
from sanic import Sanic
import os
from typing import List
try:
    from co6co_sanic_ext.cors import CORS  # The typical way to import sanic-cors
except ImportError:
    # Path hack allows examples to be run without installation.
    import os
    parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    os.sys.path.insert(0, parentdir)
    from co6co_sanic_ext.cors import CORS

def attach_cors(app:Sanic,resoutce:str=r"/v1/*",methods:List[str]=["GET", "POST", "HEAD", "OPTIONS"]):
    #跨域
    CORS_OPTIONS = {"resources": resoutce, "origins": "*", "methods": methods} 
    Extend(app, extensions=[CORS], config={"CORS": False, "CORS_OPTIONS": CORS_OPTIONS})

