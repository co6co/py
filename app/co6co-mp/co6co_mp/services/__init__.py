
from co6co_db_ext.res.result import Result
from co6co_db_ext.db_utils import QueryOneCallable
from model.enum import Account_category
from co6co_permissions.model.pos.right import AccountPO
from sqlalchemy.orm import joinedload
from sqlalchemy.sql import Select
from co6co.utils import log


async def getAccountuuid(request: Result, userOpenId: str):
    """
    通过 userOpenId 获取账号的 UUID
    """
    accountuuid = None
    try:
        queryOne = QueryOneCallable(request.ctx.session)
        select = (
            Select(AccountPO)
            .options(joinedload(AccountPO.userPO))
            .filter(AccountPO.accountName == userOpenId, AccountPO.category == Account_category.wx.val)
        )
        a: AccountPO = await queryOne(select)
        accountuuid = a.uid
    except Exception as e:
        log.warn(f"get AccountUUID error:{e}")
    return accountuuid
