from sanic import Blueprint,json
from sanic.request import Request
from datetime import datetime
from co6co_web_db.utils import JSON_util
from co6co.utils import log
 
  
signal_api = Blueprint("sigin_API")
@signal_api.signal("foo.bar.<thing>")
async def signal_handler(thing, **kwargs):
    """
    定义一个信号处理程序

    正式为:v21.6
    现在还在是测试版
    """
    print(f"[signal_handler] {thing=}", kwargs)

async def wait_for_event(app):
    while True:
        log.info("等待任务触发...")
        #用于暂停执行，直到事件被触发
        data=await signal_api.event("foo.bar.*")  
        log.info(f"该执行任务...{data}{type(data)}") 
        log.start_mark("tast state")
        print("Name:", app.m.name)
        print("PID:", app.m.pid)
        print("状态：", app.m.state)
        print("workers:", app.m.workers)
        log.end_mark("tast state") 


@signal_api.after_server_start
async def after_server_start(app, loop):
    app.add_task(wait_for_event(app))

@signal_api.route("/signal", methods=["POST","GET"]) 
async def trigger(request:Request):
   """
   触发信号-> 执行任务。
   """
   await signal_api.dispatch("foo.bar.baz")
   return JSON_util.response("Done.")
 