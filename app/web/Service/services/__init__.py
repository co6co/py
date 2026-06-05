# -*- coding: utf-8 -*-
"""
本模块设计中应不引入除 model和utils 之外的模块
"""

from re import A

from sanic import Request
from co6co.data.result import Result

from co6co_db_ext.db_utils import QueryOneCallable
from model.enum import account_category
from co6co_permissions.model.pos.right import AccountPO
from co6co_permissions.services import getSecret, generateCode
from sqlalchemy.orm import joinedload
from sqlalchemy.sql import Select
from co6co.utils import log
from sqlalchemy.ext.asyncio import AsyncSession
from model.pos.wx import tbl_WX_OpenID


async def createTicketCode(request: Result, userOpenId: str):
    """
    通过 user_openId 获取账号的 code
    通过code 换取 token
    """
    code = None
    try:
        queryOne = QueryOneCallable(request.ctx.session)
        select = (
            Select(AccountPO)
            .options(joinedload(AccountPO.userPO))
            .filter(
                AccountPO.accountName == userOpenId,
                AccountPO.category == account_category.wx.val,
            )
        )
        a: AccountPO = await queryOne(select)
        if a is not None:
            code = await generateCode(getSecret(request), a.userId, 60)
    except Exception as e:
        log.warn(f"createTicketCode error:{e}")
    return code


async def getAccountName(session: AsyncSession, userId: int):
    """
    通过 userId 获取账号的 accountName
    """
    accountName = None
    try:
        queryOne = QueryOneCallable(session)
        select = Select(AccountPO).filter(
            AccountPO.userId == userId, AccountPO.category == account_category.wx.val
        )
        a: AccountPO = await queryOne(select)
        if a is not None:
            accountName = a.accountName
        return accountName

    except Exception as e:
        log.warn(f"获取账号名失败 error:{e}")
    return accountName


async def getAppid(session: AsyncSession, userOpenId: str):
    """
    通过 微信用户的 OpenId 获取账号的 公众号 appid
    """
    appid = None
    try:
        queryOne = QueryOneCallable(session)
        select = Select(tbl_WX_OpenID).filter(tbl_WX_OpenID.openid == userOpenId)
        a: tbl_WX_OpenID = await queryOne(select)
        if a is not None:
            appid = a.appid
        return appid

    except Exception as e:
        log.warn(f"获取公众号appid error:{e}")
    return appid
