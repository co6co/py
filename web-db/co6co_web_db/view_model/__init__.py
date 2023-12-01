from functools import wraps 
from sanic.views import HTTPMethodView # 基于类的视图
from sanic import  Request
from co6co_db_ext.db_filter import absFilterItems
from co6co_db_ext.db_operations import DbPagedOperations,DbOperations,InstrumentedAttribute
from co6co_sanic_ext.model.res.result import Page_Result
from co6co_sanic_ext.utils import  JSON_util
from sqlalchemy.ext.asyncio import AsyncSession
from typing import TypeVar,Dict,List,Any
from co6co_sanic_ext.model.res.result import Result 
import aiofiles,os,multipart
from io import BytesIO



from co6co.utils import log,getDateFolder
#from api.auth import authorized

class BaseMethodView(HTTPMethodView):
    """
    视图基类： 约定 增删改查，其他未约定方法可根据实际情况具体使用
    views.POST  : --> query list
    views.PUT   :---> Add 
    view.PUT    :---> Edit
    view.DELETE :---> del

    """ 
    async def save_body(self,request:Request,root:str):
        ## 保存上传的内容 
        filePath=os.path.join(root,getDateFolder(),f"{getDateFolder(format="%Y-%m-%d-%H-%M-%S") }.data")
        filePath=os.path.abspath(filePath) # 转换为 os 所在系统路径 
        folder=os.path.dirname(filePath) 
        if not os.path.exists(folder):os.makedirs(folder)
        async with aiofiles.open(filePath, 'wb') as f:
            await f.write( request.body) 
        ## end 保存上传的内容
    async def parser_multipart_body(self,request:Request)->(Dict[str,tuple|Any],Dict[str,multipart.MultipartPart]):
        """
        解析内容: multipart/form-data; boundary=------------------------XXXXX,
        的内容
        """ 
        env={
            "REQUEST_METHOD":"POST",
            "CONTENT_LENGTH":request.headers.get("content-length"),
            "CONTENT_TYPE":request.headers.get("content-type"),
            "wsgi.input":BytesIO(request.body)
        }
        data,file=multipart.parse_form_data(env)  
        data_result={} 
        #log.info(data.__dict__)
        for key in data.__dict__.get("dict"):
            value=data.__dict__.get("dict").get(key)
            if len(value)==1: data_result.update({key:value[0]})
            else : data_result.update({key:value})
        #log.info(data_result) 
        return data_result,file

    async def save_file(self,file,path:str):
        """
        保存上传的文件
        file.name
        """
        async with aiofiles.open(path, 'wb') as f:
            await f.write(file.body) 

    async def _save_file(self,request:Request, *savePath:str,fileFieldName:str=None):
        """
        保存上传的文件
        """ 
        p_len=len(savePath)
        if fileFieldName!=None and p_len==1: 
            file = request.files.get(fileFieldName)
            await self.save_file(file,*savePath)
        elif p_len==len(request.files):
            i:int=0 
            for file in request.files: 
                file = request.files.get('file')
                await self.save_file(file,savePath[i])
                i+=1
                

    async def _get_list(self,request:Request,filterItems:absFilterItems,field:InstrumentedAttribute="*" ):
        """
        列表
        """ 
        filterItems.__dict__.update(request.json)  
        async with request.ctx.session as session:  
            opt=DbPagedOperations(session,filterItems) 
            total = await opt.get_count(field) 
            result = await opt.get_paged()   
            pageList=Page_Result.success(result)
            pageList.total=total 
            await opt.commit()
            return JSON_util.response(pageList)
    async def _del_po(self,request:Request,poType:TypeVar,pk:int ): 
        """
        删除数据库对象
        """ 
        async with request.ctx.session as session: 
            session:AsyncSession=session
            operation=DbOperations(session)
            po= await operation.get_one_by_pk(poType,pk) 
            if po==None:return JSON_util.response(Result.fail(message=f"未该'{pk}'对应得数据!"))  
            await operation.delete(po) 
            await operation.commit()    
            return JSON_util.response(Result.success())
    def getFullPath(self,root, fileName:str)->(str,str):
        """
        获取去路径和相对路径
        """
        filePath="/".join(["",getDateFolder(),fileName] ) 
        fullPath=os.path.join(root,filePath[1:])

        return fullPath,filePath
"""
class AuthMethodView(BaseMethodView): 
   decorators=[authorized] 
"""