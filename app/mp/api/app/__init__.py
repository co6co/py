from sanic import Sanic, Blueprint,Request
 
 
app_api = Blueprint("app_API")   

@app_api.main_process_start
async def main_process_start(*_):
    print(">>>>>>main_process_start_start <<<<<<")

@app_api.before_server_start
async def server_start(*_):
    print(">>>>>>before_server_start <<<<<<")
@app_api.after_server_start
async def server_started(*_):
    print(">>>>>>after_server_start <<<<<<")

@app_api.before_server_stop
async def app_before_stop(*_):
    print(">>>>>>before_server_stop <<<<<<")
@app_api.after_server_stop
async def after_server_stop(*_):
    print(">>>>>>after_server_stop <<<<<<")

@app_api.main_process_stop
async def main_process_stop(*_):
    print(">>>>>>main_process_stop <<<<<<")

'''
@app_api.middleware("request")
async def inject_session(request:Request):  
    #logger.info("mount DbSession 。。。")
    if "/nyzh/pubApi/ipncAlive" in request.path:
        print("原始数据：",request.headers,request.body)
''' 