
 
from sanic import Request 
from sqlalchemy.orm import scoped_session

from co6co_sanic_ext.utils import JSON_util
from co6co.utils import log
from model.enum.task import Task_Statue, Task_Type

from view_model import AuthMethodView
 
from co6co_sanic_ext.model.res.result import Page_Result, Result 

from model.filters.taskFilter import TaskFilterItems 
from model.pos.device import devicePo, TasksPO
import datetime,os 


class user_tasks_View(AuthMethodView):
    async def get(self, request: Request):
        """
        获取任务状态
        """
        dictList = Task_Statue.to_dict_list()
        typeList = Task_Type.to_dict_list()

        return JSON_util.response(Result.success({"statue": dictList, "types": typeList})) 
    
    async def post(self, request: Request):
        """
        列表 
        """

        param = TaskFilterItems()
        param.__dict__.update(request.json)
        param.user =self.getUserId(request)
        session: scoped_session = request.ctx.session

        result = session.execute(param.list_select)
        result = result.mappings().all()
        result = [dict(a) for a in result]

        executer = session.execute(param.count_select)
        pageList = Page_Result.success(result, total=executer.scalar())
        return JSON_util.response(pageList)


class user_task_View(AuthMethodView):
    """
    用户任务视图
    """ 
    async def delete(self, request: Request,pk:int):
        """
        删除
        """
        try: 
            userId = self.getUserId(request)
            session: scoped_session = request.ctx.session 
            po: TasksPO = session.get_one(TasksPO,pk)
            if po == None: JSON_util.response(Result.fail(message="任务不存在，请刷新重试！"))
            if po.createUser == userId:
                if po.type==Task_Type.down_task.val:
                    taskFolder = request.app.config.taskFolder
                    filePath = os.path.join(taskFolder, po.data[1:])
                    if os.path.exists(filePath):
                        os.remove(filePath)
                '''
                其他需要处理的任务
                '''
                session.delete(po) 
                session.commit() 
                return JSON_util.response(Result.success())
            else:
                return JSON_util.response(Result.fail(message="该任务不属于当前用户创建"))
        except Exception as e:
            return JSON_util.response(Result.fail(message=f"删除任务失败：{e}"))
