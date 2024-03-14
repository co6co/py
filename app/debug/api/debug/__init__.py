from sanic import Blueprint,json
from sanic.request import Request
from datetime import datetime
from co6co_web_db.utils import JSON_util
from co6co.task.thread import ThreadEvent
import time
from co6co.utils import log,isCallable

server_api = Blueprint("server_API")


async def tt():
    time.sleep(1.0)
    print("aa")
    return True
def bck():
    print("可调用对象",isCallable(tt),isCallable({}),isCallable(None))
    log.log("线程退出。。。")

@server_api.route("/debug", methods=["POST","GET"])
async def debug(request:Request):   
    header={"time:":datetime.now(),"ip":request.client_ip}
    for k in  request.headers.keys():
        header.update({k:request.headers.get(k)}) 
    t=ThreadEvent("test 线程",bck) 

    print( t.runTask(tt))
    t.close()
    return JSON_util.response(header)