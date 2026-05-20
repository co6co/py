from typing import Optional
from .view import AbsView
from .model.pos.right import UserPO
from sqlalchemy import Select
from co6co.data.result import Result, Page_Results
from co6co_db_ext.actuator import OperationOption


class changePwd_view(AbsView):
    routePath = "/changePwd"
    async def post(self):
        """
        修改密码：
        {
            oldPassword:"",
            newPassword:"",
            remark:""
        }
        """
        data = self.get_json()
        userId = self.current_user_id
        userName = self.current_user_name

        oldPassword = data["oldPassword"]
        password = data["newPassword"]
        remark = data["remark"]
        select = (Select(UserPO).filter(UserPO.userName == userName))
        one:Optional[UserPO]=await self.actuator.execute(select)
        
        if one is not None:
            if one.password != one.encrypt(oldPassword):
                return self.response_data(Result.fail(message="输入的旧密码不正确！"))
            if one.encrypt(password) == one.encrypt(oldPassword):
                return self.response_data(Result.fail(message="输入的旧密码与新密码一样！"))
            one.password = one.encrypt(password)
            if remark:
                one.remark = remark
            one.edit_assignment(userId)
        return self.response_data(Result.success())
## todo 
#resource_baseView,
class user_avatar_view(AbsView):
    routePath = "/avatar"

    async def get(self ):
        """
        获取头像
        """
        userName = self.current_user_name 
        select = Select(UserPO.avatar).filter(UserPO.userName == userName)
        avatar:Optional[str] = await self.actuator.execute( select ) 
        if avatar:
            return await self.response_local_file(request, avatar)
        else:
            svg_content = '''
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
                <!-- 头部 -->
                <circle cx="50" cy="20" r="10" fill="black"/>
                <!-- 身体 -->
                <line x1="50" y1="30" x2="50" y2="70" stroke="black" stroke-width="3"/>
                <!-- 手臂 -->
                <line x1="30" y1="40" x2="70" y2="40" stroke="black" stroke-width="3"/>
                <!-- 腿部 -->
                <line x1="50" y1="70" x2="30" y2="90" stroke="black" stroke-width="3"/>
                <line x1="50" y1="70" x2="70" y2="90" stroke="black" stroke-width="3"/>
                </svg>
            '''
            return text(svg_content, content_type='image/svg+xml')

    async def put(self ):
        """
        上传图像
        """
        userName = self.current_user_name
        select = Select(UserPO).filter(UserPO.userName == userName)
        result = await self.saveFile(request)
        if type(result) == FileResult: 
            async def edit(_, one: UserPO):
                if one != None:
                    if result.path:
                        one.avatar = result.path
                return self.response_data(Result.success(data=result.path))
            return await self.update_one(request, select, edit)
        else:
            return result

class user_info_view(AbsView):
    routePath = "/currentUser"

    async def get(self ):
        """
        当前用户信息  
        return {
            data:{
                avatar:""
                remark:""
                userName:""
            } 
        }
        """

        userName = self.current_user_name
        select = Select(UserPO.avatar, UserPO.userName, UserPO.remark).filter(UserPO.userName == userName)
        data=await  self.actuator.query_one_entity_mapping(select) 
        return self.response_data(Result.success(data))
