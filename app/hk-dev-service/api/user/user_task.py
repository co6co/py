from sanic import Sanic, Blueprint,Request
from sanic.response import json,file_stream,file
from services import authorized
from model.pos.right import UserPO
from model.pos.device import TasksPO
 
from model.filters.taskFilter  import TaskFilterItems
 
from co6co_db_ext.db_operations import DbOperations,DbPagedOperations 
from model.enum.task import Task_Statue,Task_Type 
from co6co_sanic_ext.model.res.result import Page_Result,Result
from co6co_sanic_ext.utils import JSON_util 
import os

task_api = Blueprint("task_API", url_prefix="/tasks")
@task_api.route("/list",methods=["POST",])
@authorized
async def list(request:Request):  
    """
    列表
    """
    param=TaskFilterItems()
    param.__dict__.update(request.json) 
    param.user=request.ctx.current_user["id"]
    async with request.ctx.session as session:  
        opt=DbPagedOperations(session,param) 
        total = await opt.get_count(TasksPO.id) 
        result = await opt.get_paged()   
        pageList=Page_Result.success(result)
        pageList.total=total 
        await session.commit()
        return JSON_util.response(pageList)
    
@task_api.route("/getTaskStatus",methods=["GET",])
@authorized
async def get_task_status(request:Request):
    """
    获取处理状态
    """
    dictList=Task_Statue.to_dict_list()
    typeList=Task_Type.to_dict_list()
    return JSON_util.response(Result.success({"statue": dictList,"types":typeList,"downloadTask":Task_Type.down_task.val}))

@task_api.route("/del/<pk:int>",methods=["DELETE","POST"])
@authorized
async def delete(request:Request,pk:int):
    """
    删除
    """
    userId=request.ctx.current_user["id"]
    async with request.ctx.session as session:  
        opt=DbOperations(session) 
        po:TasksPO=await opt.get_one_by_pk(TasksPO,pk)
        if po==None:JSON_util.response(Result.fail(message="任务存在请刷新重试！"))
        if po.createUser==userId:
            taskFolder=request.app.config.taskFolder
            filePath=os.path.join(taskFolder,po.data[1:] ) 
            if os.path.exists(filePath):
                os.remove(filePath)
            await opt.db_session.delete(po)
            await opt.db_session.commit()
            return JSON_util.response(Result.success())
        else:
            return JSON_util.response(Result.fail(message="该任务不属于当前用户创建"))
     
