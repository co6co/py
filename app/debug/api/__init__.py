from sanic import Blueprint,json
from sanic.request import Request
from datetime import datetime
from co6co_web_db.utils import JSON_util
 
 


server_api = Blueprint("server_API")
@server_api.route("/debug", methods=["POST","GET"])
async def debug(request:Request):   
    header={"time:":datetime.now(),"ip":request.client_ip}
    for k in  request.headers.keys():
        header.update({k:request.headers.get(k)}) 
    return JSON_util.response(header)

api = Blueprint.group(server_api, url_prefix="/rest" )
