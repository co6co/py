
from functools import wraps
from sanic import Blueprint,Sanic
from sanic.response import  json
from sanic.request import Request 
from co6co_web_db.services .jwt_service import validToken
from co6co_db_ext.res.result import Result
from co6co_sanic_ext.utils import JSON_util
from co6co.utils import log
from co6co_web_db.services.jwt_service import createToken
from sqlalchemy.ext.asyncio import AsyncSession
from co6co_db_ext.db_operations import DbOperations
from model.enum import Account_category
from model.pos.right import UserPO,AccountPO
from sqlalchemy.orm import joinedload
from sqlalchemy.sql import Select

async def getAccountuuid(request:Result ,userOpenId:str=None):  
    accountuuid=None 
    async with request.ctx.session as session:
        session:AsyncSession=session
        operation=DbOperations(session)  
        try:   
            select=(
                Select(AccountPO) 
                .options(joinedload(AccountPO.userPO))  
                .filter(AccountPO.accountName==userOpenId,AccountPO.category==Account_category.wx.val ) 
            )  
            a:AccountPO=await operation._get_one(select,False)   
            accountuuid=a.uid
        except Exception as e:
            log.err(f"创建Token失败:{e}")
        finally:
            await operation.commit() 

    return accountuuid

async def generateUserToken(request:Result,sessionId:str,data=None,userId:int=None,userOpenId:str=None,expire_seconds:int=86400): 
    token=""  
    SECRET=request.app.config.SECRET 
    if data!=None:
        token=await createToken(SECRET,data,expire_seconds)
    else: 
        async with request.ctx.session as session:
            session:AsyncSession=session
            operation=DbOperations(session)  
            
            try:  
                if userId!=None:
                    user:UserPO=await operation.get_one(UserPO,UserPO.id.__eq__(userId) )   
                    token=await createToken(SECRET,user.to_dict())   
                elif userOpenId!=None:
                    select=(
                        Select(AccountPO) 
                        .options(joinedload(AccountPO.userPO))  
                        .filter(AccountPO.accountName==userOpenId,AccountPO.category==Account_category.wx.val ) 
                    )  
                    a:AccountPO=await operation._get_one(select,False)  
                    token=await createToken(SECRET,  a.userPO.to_dict(),expire_seconds) 
                   
                   
            except Exception as e:
                log.err(f"创建Token失败:{e}")
            finally:
                await operation.commit() 

    return  { "token":token,"expireSeconds":expire_seconds,"sessionId":str(sessionId)}


def authorized(f): 
    '''
    慢慢 移走 不需要在这里
    '''
    @wraps(f)
    async def decorated_function(request, *args, **kwargs):
        # run some method that checks the request
        # for the client's authorization status
        is_authorized = await validToken(request,request.app.config.SECRET)

        if is_authorized:
            # the user is authorized.
            # run the handler method and return the response
            response = await f(request, *args, **kwargs)
            return response
        else:
            # the user is not authorized.
            return json( JSON_util().encode(Result.fail(message="not_authorized"))  , 403)
    return decorated_function 