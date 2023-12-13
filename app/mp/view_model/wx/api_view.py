
from sqlalchemy.ext.asyncio import AsyncSession
from co6co_db_ext .db_operations import DbOperations
from sanic import  Request 
from sanic.response import text,raw
from co6co_sanic_ext.utils import JSON_util,json

from view_model.wx import wx_authon_views
from model.pos.wx_where import WxMenuFilterItems,WxMenuPO
from co6co_sanic_ext.model.res.result import Result

from typing import List,Optional,Tuple
from co6co.utils import log  
from datetime import datetime
from model.enum import wx_menu_state

class config_View(wx_authon_views):
    async def get(self, request:Request): 
       """
       获取微信公众号配置的 [{openId,name}]
       """ 
       return JSON_util.response(Result.success(self.get_wx_configs(request)))

class Wx_message_View(wx_authon_views):
    """
    与页面相关，页面中不能出现 与微信 openid 相关的东西
    """
    def get(self):
        self.client.media.upload()


class WxView_Api(wx_authon_views):
    """
    群发消息
    主动设置 clientmsgid 来避免重复推送
    群发保护-->需要等待管理员进行确认
    1. 将图文消息中需要用到的图片，使用上传图文图片接口，上传成功并获得图片 URL
    2. 用户标签的群发，或对 OpenID 列表的群发，将图文消息群发出去，群发时微信会进行原创校验，并返回群发操作结果；
    3. 如果需要，还可以预览图文消息、查询群发状态，或删除已群发的消息等

    素材管理接口-->mediaID

    is_to_all=true--->使其进入公众号在微信客户端的历史消息列表【media_id 会失效，后台草稿也会被自动删除。】
    """
    async def get(self, request:Request):
       """
       未使用
       """ 
       return text("no use")
    async def post(self, request:Request):
        """
        未使用 获取列表
        """ 
        return text("no use")
    async def put(self, request:Request):
        """
        获取列表
        """ 
        return text("no use") 
    
class menus_Api(wx_authon_views):
    """
    要求: 一级菜单:max->3  字数：max 4个汉字 “...”代替
          二级菜单:max->5 字数：max 4个汉字

          刷新策略 进入公众号会话页或公众号profile页时，上一次拉取菜单的请求在5分钟以前，
    """ 
    async def get(self,request:Request): 
        return JSON_util.response(Result.success(data={"menuStates": wx_menu_state.to_dict_list()}))
    async def post(self, request:Request):
        """
        获取列表
        """ 
        param=WxMenuFilterItems() 
        return await self._get_list(request,param,WxMenuPO.id)
    
    async def put(self, request:Request):
        """
        增加菜单
        """
        po =WxMenuPO()
        po.__dict__.update(request.json)   
        current_user_id=request.ctx.current_user["id"] 
        async with request.ctx.session as session:  
            session:AsyncSession=session
            operation=DbOperations(session)
            po.id=None  
            po.createUser=current_user_id 
            operation.add_all([po])
            await session.commit()
            return JSON_util.response(Result.success()) 
    async def patch(self, request:Request):
        """
        修改菜单
        """
        return text("”") 
    
class menu_Api(wx_authon_views):
    def push_menu(self, request:Request,openId:str,content:str)->Tuple[bool,str]:
        try:
            client=self.cteate_wx_client(request,openId) 
           
            client.menu.update(json.loads(content))
            return True,""
        except Exception as e: 
            return False, str(e)
    def pull_menu(self, request:Request,openId:str)->Tuple[bool,str]:
        try:
            client=self.cteate_wx_client(request,openId) 
            result:dict=client.menu.get () 
            #obj=json.dumps( content) 
            data=result.get("menu")
            log.warn(data)
            return True,data
        except Exception as e: 
            return False, str(e)
         
    async def get(self, request:Request,pk:int):
        """
        从微信服务器获取菜单，并保存到数据库中
        """
        current_user_id=request.ctx.current_user["id"] 
        async with request.ctx.session as session:   
            operation=DbOperations(session)
            old_po:WxMenuPO= await operation.get_one_by_pk(WxMenuPO,pk)
            if old_po==None: return JSON_util.response(Result.fail(message=f"未找{pk},对应的对象!"))  
            
            f,content=self.pull_menu(request,old_po.openId) 
            if f:
                old_po.updateUser=current_user_id
                old_po.updateTime=datetime.now() 
                old_po.content=json.dumps( content,ensure_ascii=False) 
                result=Result.success()
                await operation.commit() 
            else : 
                result=Result.fail(message=content)
                await  operation.rollback() 
            return JSON_util.response(result) 
    
    async def delete(self, request:Request,pk:int):
        """
        删除数据库存储得微信菜单
        """
        return self._del_po(request,WxMenuPO,pk) 
    async def put(self, request:Request,pk:int):
        """
        更新菜单
        """
        po =WxMenuPO()
        po.__dict__.update(request.json)   
        current_user_id=request.ctx.current_user["id"] 
        async with request.ctx.session as session:   
            operation=DbOperations(session)
            old_po:WxMenuPO= await operation.get_one_by_pk(WxMenuPO,pk)
            if old_po==None: return JSON_util.response(Result.fail(message=f"未找{pk},对应的对象!"))  
            old_po.name=po.name
            old_po.openId=po.openId
            old_po.content=po.content
            old_po.updateUser=current_user_id
            old_po.updateTime=datetime.now()
            await session.commit()
            return JSON_util.response(Result.success()) 
        
     
    async def patch(self, request:Request,pk:int):
        """
        推送菜单到微信公众号
        1. 更改其他菜单状态
        2. 更改当前菜单状态为已推送
        """  
        current_user_id=request.ctx.current_user["id"] 
        async with request.ctx.session as session:   
            operation=DbOperations(session)
            old_po:WxMenuPO= await operation.get_one_by_pk(WxMenuPO,pk)
            if old_po==None: return JSON_util.response(Result.fail(message=f"未找{pk},对应的对象!"))   
            menuList:List[WxMenuPO]=await operation.get_list(WxMenuPO,WxMenuPO.openId==old_po.openId ,WxMenuPO.id !=old_po.id,remove_instance_state=False)
            for m in menuList:
                m.state=wx_menu_state.unpushed.val
                m.updateUser=current_user_id
                m.updateTime=datetime.now()
                
            old_po.updateUser=current_user_id
            old_po.updateTime=datetime.now()
            f,msg=self.push_menu(request,old_po.openId,old_po.content) 
            if f: 
                old_po.state=wx_menu_state.pushed.val
                result=Result.success()
            else :
                old_po.state=wx_menu_state.failed.val 
                result=Result.fail(message=msg)
            await session.commit() 
            return JSON_util.response(result) 
    




