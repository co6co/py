# -*- encoding:utf-8 -*-
import datetime
from sanic.response import  json
from sanic import Blueprint,Request
from sanic import exceptions
from model.pos.right import UserPO ,AccountPO ,UserGroupPO,RolePO,UserRolePO
from sqlalchemy.ext.asyncio import AsyncSession

from co6co_web_db.utils import DbJSONEncoder as JSON_util
from model.filters.UserFilterItems import UserFilterItems 
from co6co_db_ext.res.result import Result,Page_Result 

from services import authorized,generateUserToken
from co6co.utils import log
from co6co_db_ext .db_operations import DbOperations,DbPagedOperations,and_,joinedload
from sqlalchemy import func,text
from sqlalchemy.sql import Select

user_api = Blueprint("user_API", url_prefix="/user")  



@user_api.route("/ticket/<uuid:str>",methods=["POST",])
async def ticket(request:Request,uuid:str): 
    async with request.ctx.session as session:  
        session:AsyncSession=session
        select=( 
            Select(UserPO)
            .join(AccountPO,isouter=True) 
            .filter( AccountPO.uid==uuid)
        ) 
        result= await session.execute(select)
        user:UserPO=result.scalar()
        log.warn(user)
        token=await generateUserToken(request,user.to_dict())
        await session.commit()
        return  JSON_util.response(Result.success(data=token, message="登录成功")) 
    
@user_api.route("/login",methods=["POST",])
async def login(request:Request):  
    """
    登录
    """
    where =UserPO()
    where.__dict__.update(request.json) 
    async with request.ctx.session as session:  
        session:AsyncSession=session
        operation=DbOperations(session)
        log.succ(where.userName)
        user:UserPO=await operation.get_one(UserPO,UserPO.userName.__eq__(where.userName) )  
        log.err(dir(user))
        if user !=None:
            log.err(f"encode:{user.encrypt(where.password)}")
            if user.password==user.encrypt(where.password):  
                token=await generateUserToken(request,user.to_dict())
                return  JSON_util.response(Result.success(data=token, message="登录成功"))
            else :return JSON_util.response(Result.fail(message="密码不正确!"))
        return  JSON_util.response(Result.fail(message="登录用户名不存在!"))


@user_api.route("/list",methods=["POST",])
@authorized
async def list(request:Request):
    """
    列表
    """  
    param=UserFilterItems()
    param.__dict__.update(request.json) 
    async with request.ctx.session as session:  
        session:AsyncSession=session 
        opt=DbOperations(session)  
        log.start_mark("un errr")
        select=(
            Select(UserPO.id,  UserPO.state,UserPO.createTime, UserPO.userName ,UserPO.userGroupId,UserGroupPO.name.label("groupName"),UserGroupPO.code.label("groupCode")  )
            .join(UserGroupPO,isouter=True) 
            .filter(and_(*param.filter()))
            .limit(param.limit).offset(param.offset)  
        ) 
        '''
        //todo  sqlalchemy.engine.row import RowMapping   to JSON 
          '''
        result=await session.execute(select)
        result=result.mappings().all()
        result=[dict(a)  for a in  result]    
        ''' result=await session.execute(select)
        result =[dict(zip(a._fields,a))  for a in  result] 
        '''
        
        select=(
            Select( func.count( )).select_from(
                 Select(UserPO.id)
                .join(UserGroupPO,isouter=True) 
                .filter(and_(*param.filter()))
            )
        ) 
        totalResult= await session.execute(select)  
        total=totalResult.scalar_one()
        log.warn(type(result))
        
        pageResult=Page_Result.success(result ,total=total)  
        await opt.commit()
        return JSON_util.response(pageResult )

@user_api.route("/exist/<userName:str>",methods=["GET", "POST",])
@authorized
async def exist(request:Request,userName:str):  
    """
    用户名是否存在
    """
    async with request.ctx.session as session:  
        operation=DbOperations(session)
        isExist=await operation.exist(UserPO.userName==userName,column=UserPO.id)
        await session.commit()    
        if isExist:return JSON_util.response(Result.success(message=f"用户'{userName}'已存在。"))
        else: return JSON_util.response(Result.fail(  message=f"用户'{userName}'不已存在。"))

@user_api.route("/add",methods=["POST",])
@authorized
async def add(request:Request):
    """
    增加用户
    """ 
    user =UserPO()
    user.__dict__.update(request.json)   
    current_user_id=request.ctx.current_user["id"] 
    async with request.ctx.session as session:  
        session:AsyncSession=session
        operation=DbOperations(session)
        isExist=await operation.exist(UserPO.userName==user.userName,column=UserPO.id)
        if isExist:return JSON_util.response(Result.fail(message=f"增加用户'{user.userName}'已存在"))
        user.id=None 
        user.salt=user.generateSalt()  
        user.password=user.encrypt()
        user.createUser=current_user_id
        user.createTime=datetime.datetime.now()
        session.add_all([user])
        await session.commit()
        return JSON_util.response(Result.success()) 


@user_api.route("/edit/<pk:int>",methods=["POST",])
@authorized
async def edit(request:Request,pk:int): 
    """
    编辑用户
    """
    user =UserPO()  
    user.__dict__.update(request.json)  
    user.salt=user.generateSalt() 
    if user.roleId not in [1,2]:
        return JSON_util.response(Result.fail(message="选择的用户角色未知")) 
    async with request.ctx.session as session: 
        session:AsyncSession=session
        async with session.begin(): 
            operation=DbOperations(session)
            userPO= await operation.get_one_by_pk(UserPO,pk)
            # 用户名修改 
            if(user.userName !=None and user.userName !="" ):userPO.userName = user.userName
            userPO.roleId = user.roleId
            userPO.state = user.state  
            await session.commit()    
    return JSON_util.response(Result.success())


@user_api.route("/del/<pk:int>",methods=["POST","DELETE"])
@authorized
async  def delete(request:Request,pk:int):  
    """
    删除用户
    """
    if pk == 1 : return JSON_util.response(Result.fail(message= "不能删除系统默认用户！"))
    async with request.ctx.session as session: 
        session:AsyncSession=session
        operation=DbOperations(session)
        user= await operation.get_one_by_pk(UserPO,pk) 
        if user==None:
            return JSON_util.response(Result.fail(message=f"未找到该用户{pk}!"))  
        await session.delete(user) 
        await session.commit()    
        return JSON_util.response(Result.success())

@user_api.route("/currentUser")
@authorized
async def currentUser(request:Request ):
    """
    当前用户信息
    """
    user=request.ctx.current_user
    userName=user["userName"]
    async with request.ctx.session as session: 
        session:AsyncSession=session 
        operation=DbOperations(session)
        dict=await operation.get_one([UserPO.avatar,UserPO.remark],UserPO.userName==userName)
        session.commit()
        return JSON_util.response(Result.success(data=dict)) 
    
@user_api.route("/changePwd",methods=["POST"])
@authorized
async def changePwd(request:Request ):
    """
    修改当前用户密码
    """  
    data=request.json
    current_user=request.ctx.current_user 
    userName=current_user.get("userName") 
    oldPassword=data["oldPassword"]
    password=data["newPassword"]
    remark=data["remark"]
    if userName==None:return JSON_util.response(Result.fail(message="未找到当前登录用户"))
    async with request.ctx.session as session: 
        session:AsyncSession=session 
        operation=DbOperations(session)
        a1:UserPO =await operation.get_one(UserPO,UserPO.userName==userName)  
        if a1.password!=a1.encrypt(oldPassword) :JSON_util.response(Result.fail(message="密码不正确！"))
        a1.password=a1.encrypt(password)
        a1.remark=  remark if  remark!=None and remark !=None else a1.remark
        await session.commit()  
    return JSON_util.response(Result.success())

@user_api.route("/reset",methods=["POST"])
@authorized
async def reset(request:Request ): 
    """
    重置密码
    """ 
    data=request.json
    userName=data["userName"]
    password=data["password"]
    async with request.ctx.session as session: 
        session:AsyncSession=session 
        operation=DbOperations(session) 
        a1:UserPO =await operation.get_one(UserPO,UserPO.userName==userName)  
        a1.password=a1.encrypt(password)
        await session.commit()  
    return JSON_util.response(Result.success())

@user_api.route("/userList",methods=["POST","GET"])
@authorized
async def userList(request:Request ): 
    """
    用户名 list
    """  
    async with request.ctx.session as session:  
        operation=DbOperations(session) 
        userList=await operation.get_list([UserPO.userName,UserPO.id])   
        await operation.db_session.commit()
    return JSON_util.response(Result.success(data=userList))